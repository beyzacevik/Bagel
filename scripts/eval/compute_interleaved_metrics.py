"""Compute quality metrics for Bagel interleaved generation datasets.

This script compares reference frames (typically the ground-truth data)
against generated frames that follow the same Bagel parquet schema.  It
produces a report containing standard image reconstruction scores (PSNR and
SSIM) together with generation quality metrics based on feature statistics:

* Frechet Inception Distance (FID) computed on per-frame features extracted
  from an ImageNet-pretrained Inception v3 network.
* Frechet Video Distance (FVD) computed on spatio-temporal features produced
  by a Kinetics-400-pretrained 3D ResNet-18 backbone.  The implementation
  mirrors the standard FVD definition but uses the readily available
  torchvision weights to avoid additional dependencies.  Feature extraction
  can be performed in configurable batches to improve throughput when scoring
  long sequences.

The utility expects two directories filled with Bagel-compatible parquet
shards (e.g. those produced by ``convert_nuscenes_to_bagel.py``) where both
datasets share a set of identifier columns.  By default, nuScenes specific
columns (``scene_token``, ``sample_token`` and ``sample_data_token``) are
used, but any combination of columns can be supplied with ``--join-on``.

Example
-------
```bash
python scripts/eval/compute_interleaved_metrics.py \
    --reference-dir /data/bagel_nuscenes/val \
    --generated-dir /results/bagel_world_modeling/val \
    --sequence-columns scene_token sensor \
    --order-columns timestamp \
    --save-report nuscenes_val_metrics.json
```

The example above aligns samples using the nuScenes identifiers, groups
frames by ``scene_token`` and ``sensor`` for FVD, and writes the resulting
metrics to ``nuscenes_val_metrics.json``.
"""

from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, MutableMapping, Optional, Sequence, Tuple, Set

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3
from torchvision.models.inception import Inception_V3_Weights
from torchvision.models.video import R3D_18_Weights, r3d_18


DEFAULT_JOIN_KEYS = ("scene_token", "sample_token", "sample_data_token")


@dataclass
class ImagePair:
    """Container holding a matched reference/prediction pair."""

    key: Tuple[object, ...]
    reference_image: Image.Image
    generated_image: Image.Image


def _load_parquet_rows(parquet_dir: Path) -> Iterator[MutableMapping[str, object]]:
    """Yield rows from every parquet shard inside ``parquet_dir``."""

    parquet_paths = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_paths:
        raise FileNotFoundError(f"No parquet files found under '{parquet_dir}'.")

    for parquet_path in parquet_paths:
        parquet_file = pq.ParquetFile(parquet_path)
        for batch in parquet_file.iter_batches():
            columns = batch.schema.names
            arrays = [column.to_pylist() for column in batch.columns]
            for values in zip(*arrays):
                row = {name: value for name, value in zip(columns, values)}
                yield row


def _decode_image(blob: bytes) -> Image.Image:
    with Image.open(io.BytesIO(blob)) as image:
        return image.convert("RGB")


def _normalise_key(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _get_column(row: MutableMapping[str, object], column: str) -> object:
    if column not in row:
        available = ", ".join(sorted(row.keys()))
        raise KeyError(f"Column '{column}' not found in row. Available columns: {available}")
    return row[column]


def _build_key(row: MutableMapping[str, object], columns: Sequence[str]) -> Tuple[object, ...]:
    if not columns:
        raise ValueError("At least one join column must be provided.")
    return tuple(_normalise_key(_get_column(row, column)) for column in columns)


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32)
    coords -= window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()
    window_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


def compute_psnr(reference: torch.Tensor, generated: torch.Tensor) -> float:
    mse = F.mse_loss(generated, reference)
    if mse.item() == 0:
        return float("inf")
    return float(10.0 * torch.log10(1.0 / mse).item())


def compute_ssim(reference: torch.Tensor, generated: torch.Tensor, window: torch.Tensor) -> float:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    padding = window.shape[-1] // 2
    mu_ref = F.conv2d(reference.unsqueeze(0), window, padding=padding, groups=reference.shape[0])
    mu_gen = F.conv2d(generated.unsqueeze(0), window, padding=padding, groups=generated.shape[0])

    mu_ref_sq = mu_ref.pow(2)
    mu_gen_sq = mu_gen.pow(2)
    mu_ref_gen = mu_ref * mu_gen

    sigma_ref_sq = (
        F.conv2d(reference.unsqueeze(0) ** 2, window, padding=padding, groups=reference.shape[0])
        - mu_ref_sq
    )
    sigma_gen_sq = (
        F.conv2d(generated.unsqueeze(0) ** 2, window, padding=padding, groups=generated.shape[0])
        - mu_gen_sq
    )
    sigma_ref_gen = (
        F.conv2d((reference.unsqueeze(0) * generated.unsqueeze(0)), window, padding=padding, groups=reference.shape[0])
        - mu_ref_gen
    )

    numerator = (2 * mu_ref_gen + c1) * (2 * sigma_ref_gen + c2)
    denominator = (mu_ref_sq + mu_gen_sq + c1) * (sigma_ref_sq + sigma_gen_sq + c2)
    ssim_map = numerator / denominator
    return float(ssim_map.mean().item())


def _prepare_fid_model(device: torch.device) -> Tuple[torch.nn.Module, transforms.Compose]:
    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, aux_logits=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    return model, transform


def _prepare_fvd_model(device: torch.device) -> Tuple[torch.nn.Module, transforms._transforms_video.VideoClassificationPresetEval]:
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    transform = weights.transforms()
    return model, transform


def _compute_activation_statistics(features: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    features_np = features.cpu().numpy()
    mu = np.mean(features_np, axis=0)
    sigma = np.cov(features_np, rowvar=False)
    return mu, sigma


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def _resample_indices(num_frames: int, target_frames: int) -> List[int]:
    if num_frames == target_frames:
        return list(range(num_frames))
    positions = np.linspace(0, num_frames - 1, target_frames)
    return [int(round(pos)) for pos in positions]


def _prepare_video_tensor(images: Sequence[Image.Image], transform) -> torch.Tensor:
    frames = [np.array(image) for image in images]
    video = np.stack(frames)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
    target_frames = getattr(transform, "num_frames", 16)
    indices = _resample_indices(video_tensor.shape[0], target_frames)
    video_tensor = video_tensor[indices]
    video_tensor = transform(video_tensor)
    if video_tensor.ndim == 5:
        if video_tensor.shape[0] != 1:
            raise ValueError("Unexpected batch dimension returned by video transform.")
        video_tensor = video_tensor.squeeze(0)
    if video_tensor.ndim != 4:
        raise ValueError("Video transform must return a 4D tensor (C, T, H, W).")
    return video_tensor


def _process_fvd_batch(
    fvd_model: torch.nn.Module,
    device: torch.device,
    ref_batch_tensors: List[torch.Tensor],
    gen_batch_tensors: List[torch.Tensor],
    ref_feature_store: List[torch.Tensor],
    gen_feature_store: List[torch.Tensor],
) -> None:
    if not ref_batch_tensors:
        return

    ref_batch = torch.stack(ref_batch_tensors).to(device)
    gen_batch = torch.stack(gen_batch_tensors).to(device)

    ref_features = fvd_model(ref_batch).detach().cpu()
    gen_features = fvd_model(gen_batch).detach().cpu()

    ref_feature_store.extend(ref_features)
    gen_feature_store.extend(gen_features)

    ref_batch_tensors.clear()
    gen_batch_tensors.clear()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Bagel interleaved generation outputs.")
    parser.add_argument("--reference-dir", type=Path, required=True, help="Directory containing reference parquet shards.")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory containing generated parquet shards.")
    parser.add_argument(
        "--join-on",
        nargs="+",
        default=DEFAULT_JOIN_KEYS,
        help="Columns used to align reference and generated samples.",
    )
    parser.add_argument(
        "--sequence-columns",
        nargs="+",
        default=("scene_token",),
        help="Columns that define a video sequence for FVD computation.",
    )
    parser.add_argument(
        "--order-columns",
        nargs="+",
        default=("timestamp",),
        help="Columns used to sort frames within a sequence.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on the number of matched samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for FID feature extraction.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device (cpu or cuda).",
    )
    parser.add_argument(
        "--fvd-batch-size",
        type=int,
        default=4,
        help="Number of sequences to process per FVD forward pass.",
    )
    parser.add_argument("--save-report", type=Path, default=None, help="Optional path to store the aggregated metrics in JSON format.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    reference_index: Dict[Tuple[object, ...], Image.Image] = {}
    order_lookup: Dict[Tuple[object, ...], Tuple[object, ...]] = {}
    sequence_lookup: Dict[Tuple[object, ...], Tuple[object, ...]] = {}

    for row in _load_parquet_rows(args.reference_dir):
        key = _build_key(row, args.join_on)
        reference_index[key] = _decode_image(bytes(_get_column(row, "image")))
        order_lookup[key] = tuple(_get_column(row, column) for column in args.order_columns)
        sequence_lookup[key] = tuple(_get_column(row, column) for column in args.sequence_columns)

    if not reference_index:
        raise RuntimeError("No samples found in the reference dataset.")

    matched_pairs: List[ImagePair] = []
    matched_keys: Set[Tuple[object, ...]] = set()
    missing_references: Set[Tuple[object, ...]] = set()

    generated_sequences: Dict[Tuple[object, ...], List[Tuple[Tuple[object, ...], Image.Image]]] = defaultdict(list)
    reference_sequences: Dict[Tuple[object, ...], List[Tuple[Tuple[object, ...], Image.Image]]] = defaultdict(list)

    for key, image in reference_index.items():
        seq_key = sequence_lookup[key]
        order_key = order_lookup[key]
        reference_sequences[seq_key].append((order_key, image))

    for row in _load_parquet_rows(args.generated_dir):
        key = _build_key(row, args.join_on)
        generated_image = _decode_image(bytes(_get_column(row, "image")))
        order_key = tuple(_get_column(row, column) for column in args.order_columns)
        seq_key = tuple(_get_column(row, column) for column in args.sequence_columns)
        generated_sequences[seq_key].append((order_key, generated_image))

        if key not in reference_index:
            missing_references.add(key)
            continue

        matched_pairs.append(ImagePair(key=key, reference_image=reference_index[key], generated_image=generated_image))
        matched_keys.add(key)

        if args.max_samples is not None and len(matched_pairs) >= args.max_samples:
            break

    if not matched_pairs:
        raise RuntimeError("No overlapping samples found between reference and generated datasets.")

    window = _gaussian_window(window_size=11, sigma=1.5, channels=3).to(device)
    to_tensor = transforms.ToTensor()

    psnr_values: List[float] = []
    ssim_values: List[float] = []

    fid_model, fid_transform = _prepare_fid_model(device)
    fid_ref_features: List[torch.Tensor] = []
    fid_gen_features: List[torch.Tensor] = []

    with torch.no_grad():
        for batch_start in range(0, len(matched_pairs), args.batch_size):
            batch_pairs = matched_pairs[batch_start : batch_start + args.batch_size]

            ref_batch_tensors = []
            gen_batch_tensors = []
            for pair in batch_pairs:
                ref_tensor = to_tensor(pair.reference_image).to(device)
                gen_tensor = to_tensor(pair.generated_image).to(device)
                psnr_values.append(compute_psnr(ref_tensor, gen_tensor))
                ssim_values.append(compute_ssim(ref_tensor, gen_tensor, window))

                ref_batch_tensors.append(fid_transform(pair.reference_image))
                gen_batch_tensors.append(fid_transform(pair.generated_image))

            ref_batch = torch.stack(ref_batch_tensors).to(device)
            gen_batch = torch.stack(gen_batch_tensors).to(device)

            fid_ref_features.append(fid_model(ref_batch))
            fid_gen_features.append(fid_model(gen_batch))

    ref_activations = torch.cat(fid_ref_features, dim=0)
    gen_activations = torch.cat(fid_gen_features, dim=0)

    mu_ref, sigma_ref = _compute_activation_statistics(ref_activations)
    mu_gen, sigma_gen = _compute_activation_statistics(gen_activations)
    fid_score = _frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)

    fvd_model, fvd_transform = _prepare_fvd_model(device)

    video_ref_features: List[torch.Tensor] = []
    video_gen_features: List[torch.Tensor] = []
    fvd_ref_batch: List[torch.Tensor] = []
    fvd_gen_batch: List[torch.Tensor] = []

    shared_sequences = set(reference_sequences.keys()) & set(generated_sequences.keys())

    with torch.no_grad():
        for seq_key in sorted(shared_sequences):
            ref_frames = [image for _, image in sorted(reference_sequences[seq_key], key=lambda item: item[0])]
            gen_frames = [image for _, image in sorted(generated_sequences[seq_key], key=lambda item: item[0])]

            if len(ref_frames) < 2 or len(gen_frames) < 2:
                continue

            try:
                ref_tensor = _prepare_video_tensor(ref_frames, fvd_transform)
                gen_tensor = _prepare_video_tensor(gen_frames, fvd_transform)
            except ValueError:
                continue

            fvd_ref_batch.append(ref_tensor)
            fvd_gen_batch.append(gen_tensor)

            if len(fvd_ref_batch) >= args.fvd_batch_size:
                _process_fvd_batch(
                    fvd_model,
                    device,
                    fvd_ref_batch,
                    fvd_gen_batch,
                    video_ref_features,
                    video_gen_features,
                )

        _process_fvd_batch(
            fvd_model,
            device,
            fvd_ref_batch,
            fvd_gen_batch,
            video_ref_features,
            video_gen_features,
        )

    if video_ref_features and video_gen_features:
        ref_video_activations = torch.stack(video_ref_features)
        gen_video_activations = torch.stack(video_gen_features)

        mu_ref_video, sigma_ref_video = _compute_activation_statistics(ref_video_activations)
        mu_gen_video, sigma_gen_video = _compute_activation_statistics(gen_video_activations)
        fvd_score = _frechet_distance(mu_ref_video, sigma_ref_video, mu_gen_video, sigma_gen_video)
    else:
        fvd_score = float("nan")

    limit_reached = args.max_samples is not None and len(matched_pairs) >= args.max_samples

    psnr_array = np.array(psnr_values, dtype=np.float64)
    psnr_array = np.where(np.isfinite(psnr_array), psnr_array, 100.0)
    ssim_array = np.clip(np.array(ssim_values, dtype=np.float64), -1.0, 1.0)

    metrics = {
        "num_pairs": len(matched_pairs),
        "num_sequences": len(shared_sequences),
        "psnr": float(psnr_array.mean()),
        "ssim": float(ssim_array.mean()),
        "fid": fid_score,
        "fvd": fvd_score,
        "missing_predictions": int(len(reference_index) - len(matched_keys)),
        "missing_references": len(missing_references),
        "max_samples": args.max_samples,
        "max_samples_limit_reached": limit_reached,
    }

    print(json.dumps(metrics, indent=2))

    if args.save_report is not None:
        args.save_report.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
