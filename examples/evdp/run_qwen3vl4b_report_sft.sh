#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AUTODL_TMP_ROOT="${AUTODL_TMP_ROOT:-/root/autodl-tmp}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${AUTODL_TMP_ROOT}/ms-swift}"
CACHE_ROOT="${CACHE_ROOT:-${TRAIN_RUN_ROOT}/cache}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${TRAIN_RUN_ROOT}/logs/qwen3vl4b_mimic_report_sft}"
TMPDIR="${TMPDIR:-${TRAIN_RUN_ROOT}/tmp}"
HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${CACHE_ROOT}/modelscope}"
TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${CACHE_ROOT}/nv_cuda}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"

USE_UV="${USE_UV:-1}"
if [[ "${USE_UV}" == "1" ]]; then
  PYTHON_CMD=(uv run python)
  SWIFT_CMD=(uv run swift)
else
  PYTHON_CMD=(python)
  SWIFT_CMD=(swift)
fi

MIMIC_ROOT="${MIMIC_ROOT:-/root/autodl-tmp/mimic-cxr-jpg/2.1.0}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/mimic_cxr_report_sft}"
TRAIN_JSONL="${TRAIN_JSONL:-${DATA_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${DATA_DIR}/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_RUN_ROOT}/output/qwen3vl4b_mimic_report_sft}"
SWIFT_LOG_FILE="${SWIFT_LOG_FILE:-${RUN_LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
KEEP_OUTPUT_DIR_ON_RESUME="${KEEP_OUTPUT_DIR_ON_RESUME:-0}"

PREPARE_DATA="${PREPARE_DATA:-0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-}"
MAX_IMAGES_PER_STUDY="${MAX_IMAGES_PER_STUDY:-2}"

MODEL="${MODEL:-Qwen/Qwen3-VL-4B-Instruct}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-1024}"
MAX_PIXELS="${MAX_PIXELS:-1003520}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
  OMP_NUM_THREADS=1
fi

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
EVAL_STEPS="${EVAL_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
PADDING_FREE="${PADDING_FREE:-true}"
PACKING="${PACKING:-false}"
DEEPSPEED="${DEEPSPEED:-}"
REPORT_TO="${REPORT_TO:-none}"
DRY_RUN="${DRY_RUN:-0}"
REQUIRE_FLASH_ATTN="${REQUIRE_FLASH_ATTN:-0}"

find_latest_checkpoint() {
  local search_dir="$1"
  if [[ ! -d "${search_dir}" ]]; then
    return 1
  fi
  find "${search_dir}" -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

if [[ "${PREPARE_DATA}" == "1" || ! -s "${TRAIN_JSONL}" ]]; then
  PREP_ARGS=(
    "${SCRIPT_DIR}/prepare_mimic_cxr_report_sft.py"
    --mimic-root "${MIMIC_ROOT}"
    --output-dir "${DATA_DIR}"
    --max-images-per-study "${MAX_IMAGES_PER_STUDY}"
  )
  if [[ -n "${MAX_TRAIN_SAMPLES}" ]]; then
    PREP_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_VAL_SAMPLES}" ]]; then
    PREP_ARGS+=(--max-val-samples "${MAX_VAL_SAMPLES}")
  fi
  if [[ -n "${MAX_TEST_SAMPLES}" ]]; then
    PREP_ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  fi
  "${PYTHON_CMD[@]}" "${PREP_ARGS[@]}"
fi

if [[ ! -s "${TRAIN_JSONL}" ]]; then
  echo "Training JSONL not found or empty: ${TRAIN_JSONL}" >&2
  exit 1
fi

if [[ "${ATTN_IMPL}" == "flash_attn" || "${ATTN_IMPL}" == "flash_attention_2" ]]; then
  if ! "${PYTHON_CMD[@]}" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('flash_attn') else 1)" >/dev/null 2>&1; then
    if [[ "${REQUIRE_FLASH_ATTN}" == "1" ]]; then
      echo "flash_attn was requested but the Python package is not importable." >&2
      exit 1
    fi
    echo "flash_attn is not importable; falling back to ATTN_IMPL=sdpa PADDING_FREE=false PACKING=false." >&2
    ATTN_IMPL=sdpa
    PADDING_FREE=false
    PACKING=false
  fi
fi

RESUME_ARGS=()
ADD_VERSION_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  if [[ "${RESUME_FROM_CHECKPOINT}" == "latest" || "${RESUME_FROM_CHECKPOINT}" == "auto" ]]; then
    LATEST_CHECKPOINT="$(find_latest_checkpoint "${OUTPUT_DIR}")"
    if [[ -z "${LATEST_CHECKPOINT}" ]]; then
      echo "No checkpoint found under OUTPUT_DIR=${OUTPUT_DIR}" >&2
      exit 1
    fi
    RESUME_FROM_CHECKPOINT="${LATEST_CHECKPOINT}"
  fi
  if [[ ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "RESUME_FROM_CHECKPOINT does not exist or is not a directory: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 1
  fi
  if [[ "${KEEP_OUTPUT_DIR_ON_RESUME}" != "1" ]]; then
    OUTPUT_DIR="$(dirname "${RESUME_FROM_CHECKPOINT}")"
  fi
  RESUME_ARGS=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  ADD_VERSION_ARGS=(--add_version false)
fi

VAL_ARGS=()
if [[ -s "${VAL_JSONL}" ]]; then
  VAL_ARGS=(--val_dataset "${VAL_JSONL}")
else
  VAL_ARGS=(--split_dataset_ratio 0.01)
fi

DS_ARGS=()
if [[ -n "${DEEPSPEED}" ]]; then
  DS_ARGS=(--deepspeed "${DEEPSPEED}")
fi

mkdir -p \
  "${OUTPUT_DIR}" \
  "${RUN_LOG_DIR}" \
  "${TMPDIR}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" \
  "${TORCH_EXTENSIONS_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${XDG_CACHE_HOME}"

export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE
export IMAGE_MAX_TOKEN_NUM
export MAX_PIXELS
export PYTORCH_CUDA_ALLOC_CONF
export OMP_NUM_THREADS
export TMPDIR
export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export MODELSCOPE_CACHE
export TORCH_HOME
export TORCH_EXTENSIONS_DIR
export TRITON_CACHE_DIR
export CUDA_CACHE_PATH
export XDG_CACHE_HOME

SFT_CMD=(
  "${SWIFT_CMD[@]}" sft
  --model "${MODEL}"
  --dataset "${TRAIN_JSONL}"
  "${VAL_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  --load_from_cache_file true
  --tuner_type lora
  --torch_dtype bfloat16
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}"
  --attn_impl "${ATTN_IMPL}"
  --padding_free "${PADDING_FREE}"
  --packing "${PACKING}"
  --learning_rate "${LEARNING_RATE}"
  --lora_rank "${LORA_RANK}"
  --lora_alpha "${LORA_ALPHA}"
  --target_modules all-linear
  --freeze_vit true
  --freeze_aligner true
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --gradient_checkpointing true
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --save_total_limit "${SAVE_TOTAL_LIMIT}"
  --logging_steps "${LOGGING_STEPS}"
  --max_length "${MAX_LENGTH}"
  --output_dir "${OUTPUT_DIR}"
  "${ADD_VERSION_ARGS[@]}"
  --warmup_ratio "${WARMUP_RATIO}"
  --dataset_num_proc "${DATASET_NUM_PROC}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --report_to "${REPORT_TO}"
  "${DS_ARGS[@]}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%q NPROC_PER_NODE=%q IMAGE_MAX_TOKEN_NUM=%q MAX_PIXELS=%q ' \
    "${CUDA_VISIBLE_DEVICES}" "${NPROC_PER_NODE}" "${IMAGE_MAX_TOKEN_NUM}" "${MAX_PIXELS}"
  printf 'OMP_NUM_THREADS=%q TMPDIR=%q HF_HOME=%q HF_DATASETS_CACHE=%q TRANSFORMERS_CACHE=%q ' \
    "${OMP_NUM_THREADS}" "${TMPDIR}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TRANSFORMERS_CACHE}"
  printf 'MODELSCOPE_CACHE=%q TORCH_EXTENSIONS_DIR=%q TRITON_CACHE_DIR=%q CUDA_CACHE_PATH=%q SWIFT_LOG_FILE=%q ' \
    "${MODELSCOPE_CACHE}" "${TORCH_EXTENSIONS_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${SWIFT_LOG_FILE}"
  printf '%q ' "${SFT_CMD[@]}"
  printf '\n'
  exit 0
fi

echo "Writing training stdout/stderr to: ${SWIFT_LOG_FILE}" >&2
set +e
"${SFT_CMD[@]}" 2>&1 | tee -a "${SWIFT_LOG_FILE}"
status=${PIPESTATUS[0]}
set -e
exit "${status}"
