# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ -z "${REFERENCE_DIR:-}" ]]; then
  echo "REFERENCE_DIR must point to the ground-truth Bagel parquet directory." >&2
  exit 1
fi

if [[ -z "${GENERATED_DIR:-}" ]]; then
  echo "GENERATED_DIR must point to the generated Bagel parquet directory." >&2
  exit 1
fi

DEVICE=${DEVICE:-$(python -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}
BATCH_SIZE=${BATCH_SIZE:-32}
FVD_BATCH_SIZE=${FVD_BATCH_SIZE:-4}
MAX_SAMPLES_ARG=${MAX_SAMPLES:-}

CLI_ARGS=(
  --reference-dir "${REFERENCE_DIR}"
  --generated-dir "${GENERATED_DIR}"
  --batch-size "${BATCH_SIZE}"
  --fvd-batch-size "${FVD_BATCH_SIZE}"
  --device "${DEVICE}"
)

if [[ -n "${JOIN_ON:-}" ]]; then
  # shellcheck disable=SC2206
  CLI_ARGS+=(--join-on ${JOIN_ON})
fi

if [[ -n "${SEQUENCE_COLUMNS:-}" ]]; then
  # shellcheck disable=SC2206
  CLI_ARGS+=(--sequence-columns ${SEQUENCE_COLUMNS})
fi

if [[ -n "${ORDER_COLUMNS:-}" ]]; then
  # shellcheck disable=SC2206
  CLI_ARGS+=(--order-columns ${ORDER_COLUMNS})
fi

if [[ -n "${MAX_SAMPLES_ARG}" ]]; then
  CLI_ARGS+=(--max-samples "${MAX_SAMPLES_ARG}")
fi

if [[ -n "${SAVE_REPORT:-}" ]]; then
  CLI_ARGS+=(--save-report "${SAVE_REPORT}")
fi

python scripts/eval/compute_interleaved_metrics.py "${CLI_ARGS[@]}"
