"""Convert the nuScenes dataset into the Bagel parquet format.

This utility iterates over nuScenes scenes, extracts camera frames and
associated metadata, and stores them in the Bagel T2I parquet format.  The
resulting parquet files contain binary encoded images together with a JSON
string under the ``captions`` column which mirrors the format consumed by the
Bagel training and evaluation pipelines.

Example
-------
```bash
python scripts/convert_nuscenes_to_bagel.py \
    --nuscenes-root /data/nuscenes \
    --version v1.0-trainval \
    --splits train val \
    --output-dir /data/bagel_nuscenes \
    --sensors CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
    --shard-size 512
```

The command above will create ``/data/bagel_nuscenes/train`` and
``/data/bagel_nuscenes/val`` directories filled with parquet shards ready to be
consumed by Bagel's dataloaders.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

try:  # ``nuscenes-devkit`` is an optional dependency for this script.
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
except ImportError as exc:  # pragma: no cover - clearly inform the user.
    raise SystemExit(
        "nuscenes-devkit is required to run this conversion script. "
        "Install it via `pip install nuscenes-devkit`."
    ) from exc


@dataclass
class ConversionStats:
    """Metadata collected while converting a single split."""

    num_samples: int = 0
    num_files: int = 0
    parquet_rows: MutableMapping[str, int] = None  # type: ignore[assignment]
    parquet_row_groups: MutableMapping[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.parquet_rows = {}
        self.parquet_row_groups = {}

    def register_file(self, file_path: Path, num_rows: int, num_row_groups: int) -> None:
        relative_path = str(file_path)
        self.parquet_rows[relative_path] = num_rows
        self.parquet_row_groups[relative_path] = num_row_groups
        self.num_samples += num_rows
        self.num_files += 1


def _normalise_split_name(split: str) -> str:
    split = split.strip().lower()
    if split in {"train", "val", "test"}:
        return split
    if split in {"mini_train", "mini_val", "mini_test"}:
        return split
    if split in {"train_detect", "val_detect", "train_track", "val_track"}:
        return split
    raise ValueError(f"Unsupported nuScenes split: {split}")


def _load_split_scenes(version: str, requested_split: str) -> Sequence[str]:
    split_lookup = create_splits_scenes()
    split_key = _normalise_split_name(requested_split)

    if version == "v1.0-mini" and not split_key.startswith("mini"):
        raise ValueError(
            f"Version '{version}' only contains the mini splits; use mini_train/mini_val/mini_test instead of '{requested_split}'."
        )

    if version != "v1.0-mini" and split_key.startswith("mini"):
        raise ValueError(
            f"Split '{requested_split}' is only available for version v1.0-mini."
        )

    if split_key not in split_lookup:
        raise KeyError(f"nuScenes split '{requested_split}' not found in the split definitions.")

    return split_lookup[split_key]


def _summarise_annotations(nusc: NuScenes, sample_record: Mapping[str, object]) -> Counter:
    """Return a counter with category counts for all annotations in a sample."""

    counts: Counter = Counter()
    for ann_token in sample_record["anns"]:  # type: ignore[index]
        ann_rec = nusc.get("sample_annotation", ann_token)
        category_name: str = ann_rec["category_name"]
        primary_category = category_name.split(".")[0].replace("_", " ")
        counts[primary_category] += 1
    return counts


def _format_category_text(category_counts: Counter) -> Optional[str]:
    if not category_counts:
        return None
    ordered = sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
    parts = [f"{name} ({count})" for name, count in ordered]
    return ", ".join(parts)


def _build_caption(
    scene_record: Mapping[str, object],
    log_record: Mapping[str, object],
    sensor_name: str,
    capture_time: datetime,
    category_counts: Counter,
) -> str:
    """Create a human readable caption for a sample.

    The caption intentionally mirrors the simple JSON based captions used in the
    standard Bagel text-to-image datasets.
    """

    description = (scene_record.get("description") or "").strip()  # type: ignore[arg-type]
    location = (log_record.get("location") or "").strip()  # type: ignore[arg-type]

    fragments: List[str] = []
    if description:
        fragments.append(description.rstrip("."))

    fragments.append(
        f"Camera {sensor_name.replace('_', ' ')} captured an autonomous driving scene"
    )

    if location:
        fragments[-1] += f" in {location}"

    fragments[-1] += f" on {capture_time.strftime('%Y-%m-%d %H:%M:%S %Z')}"

    category_text = _format_category_text(category_counts)
    if category_text:
        fragments.append(f"Visible objects include {category_text}.")
    else:
        fragments.append("The frame does not contain labelled dynamic objects.")

    return " ".join(fragments)


def _create_metadata(
    scene_record: Mapping[str, object],
    sample_record: Mapping[str, object],
    sample_data_record: Mapping[str, object],
    sensor_name: str,
    category_counts: Counter,
) -> Dict[str, object]:
    log_token = scene_record.get("log_token")
    ego_pose_token = sample_data_record.get("ego_pose_token")
    calibrated_sensor_token = sample_data_record.get("calibrated_sensor_token")

    metadata: Dict[str, object] = {
        "scene_token": scene_record["token"],
        "scene_name": scene_record["name"],
        "sample_token": sample_record["token"],
        "sample_data_token": sample_data_record["token"],
        "sensor": sensor_name,
        "timestamp": sample_data_record["timestamp"],
        "num_labeled_objects": sum(category_counts.values()),
    }

    if log_token is not None:
        metadata["log_token"] = log_token
    if ego_pose_token is not None:
        metadata["ego_pose_token"] = ego_pose_token
    if calibrated_sensor_token is not None:
        metadata["calibrated_sensor_token"] = calibrated_sensor_token

    return metadata


def _iter_samples(
    nusc: NuScenes,
    scene_names: Sequence[str],
    sensors: Sequence[str],
    verbose: bool,
) -> Iterator[Tuple[Mapping[str, object], Mapping[str, object], Mapping[str, object], str]]:
    """Yield tuples with scene, sample, and sample_data records for the requested sensors."""

    scene_lookup: Dict[str, Mapping[str, object]] = {scene["name"]: scene for scene in nusc.scene}

    for scene_name in scene_names:
        scene_record = scene_lookup.get(scene_name)
        if scene_record is None:
            if verbose:
                print(f"[WARN] Scene '{scene_name}' not found in the nuScenes metadata. Skipping.")
            continue

        log_record = nusc.get("log", scene_record["log_token"])

        sample_token = scene_record["first_sample_token"]
        while sample_token:
            sample_record = nusc.get("sample", sample_token)
            for sensor_name in sensors:
                data_token = sample_record["data"].get(sensor_name)  # type: ignore[index]
                if not data_token:
                    if verbose:
                        print(
                            f"[WARN] Sample {sample_record['token']} does not contain sensor {sensor_name}."
                        )
                    continue
                sample_data_record = nusc.get("sample_data", data_token)
                yield scene_record, log_record, sample_record, sample_data_record, sensor_name

            sample_token = sample_record["next"]


def _write_shard(
    records: List[Dict[str, object]],
    schema: pa.Schema,
    destination: Path,
    shard_prefix: str,
    shard_index: int,
    row_group_size: Optional[int],
    stats: ConversionStats,
) -> None:
    if not records:
        return

    table = pa.Table.from_pylist(records, schema=schema)
    file_name = f"{shard_prefix}_{shard_index:05d}.parquet"
    file_path = destination / file_name
    pq.write_table(table, file_path, compression="zstd", row_group_size=row_group_size)

    parquet_file = pq.ParquetFile(file_path)
    stats.register_file(file_path, parquet_file.metadata.num_rows, parquet_file.metadata.num_row_groups)


def convert_split(
    nusc: NuScenes,
    output_dir: Path,
    split: str,
    sensors: Sequence[str],
    shard_size: int,
    row_group_size: Optional[int],
    verbose: bool,
) -> ConversionStats:
    scene_names = _load_split_scenes(nusc.version, split)
    stats = ConversionStats()

    schema = pa.schema(
        [
            ("image", pa.binary()),
            ("captions", pa.string()),
            ("scene_token", pa.string()),
            ("scene_name", pa.string()),
            ("sample_token", pa.string()),
            ("sample_data_token", pa.string()),
            ("sensor", pa.string()),
            ("timestamp", pa.int64()),
            ("metadata", pa.string()),
        ]
    )

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    shard_records: List[Dict[str, object]] = []
    shard_index = 0

    for scene_record, log_record, sample_record, sample_data_record, sensor_name in _iter_samples(
        nusc, scene_names, sensors, verbose
    ):
        image_path = Path(nusc.get_sample_data_path(sample_data_record["token"]))
        try:
            image_bytes = image_path.read_bytes()
        except FileNotFoundError:
            if verbose:
                print(f"[WARN] Image file not found: {image_path}")
            continue
        except OSError as exc:
            if verbose:
                print(f"[WARN] Failed to read image '{image_path}': {exc}")
            continue

        category_counts = _summarise_annotations(nusc, sample_record)
        timestamp = int(sample_data_record["timestamp"])
        capture_time = datetime.fromtimestamp(timestamp / 1_000_000, tz=timezone.utc)
        caption = _build_caption(scene_record, log_record, sensor_name, capture_time, category_counts)
        metadata = _create_metadata(scene_record, sample_record, sample_data_record, sensor_name, category_counts)

        shard_records.append(
            {
                "image": image_bytes,
                "captions": json.dumps({"caption": caption}, ensure_ascii=False),
                "scene_token": scene_record["token"],
                "scene_name": scene_record["name"],
                "sample_token": sample_record["token"],
                "sample_data_token": sample_data_record["token"],
                "sensor": sensor_name,
                "timestamp": timestamp,
                "metadata": json.dumps(metadata, ensure_ascii=False),
            }
        )

        if len(shard_records) >= shard_size:
            _write_shard(shard_records, schema, split_dir, f"{split}", shard_index, row_group_size, stats)
            shard_records.clear()
            shard_index += 1

    if shard_records:
        _write_shard(shard_records, schema, split_dir, f"{split}", shard_index, row_group_size, stats)

    return stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert nuScenes to Bagel parquet format.")
    parser.add_argument(
        "--nuscenes-root",
        required=True,
        type=Path,
        help="Path to the root directory of the nuScenes dataset (containing the 'samples' folder).",
    )
    parser.add_argument(
        "--version",
        required=True,
        choices=[
            "v1.0-trainval",
            "v1.0-test",
            "v1.0-mini",
            "v1.0-trainval_meta",
            "v1.0-test_meta",
        ],
        help="nuScenes version string to load.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="nuScenes splits to convert (e.g. train val). Defaults to the standard splits for the chosen version.",
    )
    parser.add_argument(
        "--sensors",
        nargs="+",
        default=(
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ),
        help="Camera sensors to include in the export.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where the Bagel formatted parquet files will be stored.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1024,
        help="Number of samples per parquet shard.",
    )
    parser.add_argument(
        "--row-group-size",
        type=int,
        default=None,
        help="Optional Parquet row group size. Defaults to the shard size (resulting in one row group per shard).",
    )
    parser.add_argument(
        "--parquet-info-path",
        type=Path,
        default=None,
        help="Optional path to store a JSON file with parquet metadata (num_rows & num_row_groups).",
    )
    parser.add_argument(
        "--dataset-info-path",
        type=Path,
        default=None,
        help=(
            "Optional path to store a dataset info JSON compatible with Bagel's ``DATASET_INFO`` format "
            "for the processed splits."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit verbose logging during conversion.",
    )

    args = parser.parse_args(argv)
    if args.row_group_size is None:
        args.row_group_size = args.shard_size

    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    nusc = NuScenes(version=args.version, dataroot=str(args.nuscenes_root), verbose=args.verbose)

    splits = args.splits
    if not splits:
        if args.version == "v1.0-mini":
            splits = ["mini_train", "mini_val"]
        elif args.version == "v1.0-trainval":
            splits = ["train", "val"]
        elif args.version == "v1.0-test":
            splits = ["test"]
        elif args.version == "v1.0-trainval_meta":
            splits = ["train", "val"]
        elif args.version == "v1.0-test_meta":
            splits = ["test"]
        else:  # pragma: no cover - defensive fallback, shouldn't happen due to argparse choices.
            raise ValueError(f"Unhandled default splits for version '{args.version}'.")

    dataset_info: Dict[str, Dict[str, object]] = {}
    parquet_info: Dict[str, Dict[str, int]] = {}

    for split in splits:
        if args.verbose:
            print(f"[INFO] Converting split '{split}'...")
        stats = convert_split(
            nusc=nusc,
            output_dir=output_dir,
            split=split,
            sensors=args.sensors,
            shard_size=args.shard_size,
            row_group_size=args.row_group_size,
            verbose=args.verbose,
        )
        if args.verbose:
            print(
                f"[INFO] Finished split '{split}': {stats.num_samples} samples across {stats.num_files} parquet files."
            )

        dataset_info[split] = {
            "data_dir": str((output_dir / split).resolve()),
            "num_files": stats.num_files,
            "num_total_samples": stats.num_samples,
        }
        parquet_info.update(
            {
                path: {"num_rows": stats.parquet_rows[path], "num_row_groups": stats.parquet_row_groups[path]}
                for path in stats.parquet_rows
            }
        )

    if args.dataset_info_path:
        args.dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
        with args.dataset_info_path.open("w", encoding="utf-8") as fh:
            json.dump(dataset_info, fh, indent=2)
        if args.verbose:
            print(f"[INFO] Wrote dataset info to {args.dataset_info_path}")

    if args.parquet_info_path:
        args.parquet_info_path.parent.mkdir(parents=True, exist_ok=True)
        with args.parquet_info_path.open("w", encoding="utf-8") as fh:
            json.dump(parquet_info, fh, indent=2)
        if args.verbose:
            print(f"[INFO] Wrote parquet metadata to {args.parquet_info_path}")


if __name__ == "__main__":
    main()
