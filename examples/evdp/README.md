# EVDP MIMIC-CXR Report-Level SFT

This folder contains a minimal report-level Direct-SFT entry for
`Qwen/Qwen3-VL-4B-Instruct` using ms-swift.

## Build JSONL

```bash
cd /root/ms-swift
uv run python examples/evdp/prepare_mimic_cxr_report_sft.py \
  --mimic-root /root/autodl-tmp/mimic-cxr-jpg/2.1.0 \
  --output-dir examples/evdp/data/mimic_cxr_report_sft \
  --max-images-per-study 2
```

For a smoke dataset:

```bash
uv run python examples/evdp/prepare_mimic_cxr_report_sft.py \
  --mimic-root /root/autodl-tmp/mimic-cxr-jpg/2.1.0 \
  --output-dir examples/evdp/data/mimic_cxr_report_sft_smoke \
  --max-train-samples 32 \
  --max-val-samples 8 \
  --max-test-samples 8
```

The generated files are standard ms-swift multimodal SFT JSONL:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `summary.json`

Each sample is study-level. The input contains one or more `<image>` tokens and
absolute image paths, and the target is the cleaned Findings/Impression report.

## Storage Layout

The launch script keeps training artifacts under `/root/autodl-tmp` by default:

- checkpoints, trainer state, and `logging.jsonl`:
  `/root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft`
- stdout/stderr run logs:
  `/root/autodl-tmp/ms-swift/logs/qwen3vl4b_mimic_report_sft`
- runtime cache and temporary files:
  `/root/autodl-tmp/ms-swift/cache` and `/root/autodl-tmp/ms-swift/tmp`

Useful storage overrides:

```bash
AUTODL_TMP_ROOT=/root/autodl-tmp \
TRAIN_RUN_ROOT=/root/autodl-tmp/ms-swift \
OUTPUT_DIR=/root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft_debug \
RUN_LOG_DIR=/root/autodl-tmp/ms-swift/logs/qwen3vl4b_mimic_report_sft_debug \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

To inspect the final training command without launching training:

```bash
DRY_RUN=1 \
DATA_DIR=examples/evdp/data/mimic_cxr_report_sft_smoke \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

## Train Qwen3-VL-4B

```bash
cd /root/ms-swift
PREPARE_DATA=1 \
MIMIC_ROOT=/root/autodl-tmp/mimic-cxr-jpg/2.1.0 \
DATA_DIR=examples/evdp/data/mimic_cxr_report_sft \
CUDA_VISIBLE_DEVICES=0 \
USE_UV=1 \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

Useful overrides:

```bash
MAX_TRAIN_SAMPLES=1024 \
MAX_VAL_SAMPLES=128 \
MAX_IMAGES_PER_STUDY=2 \
MAX_LENGTH=4096 \
IMAGE_MAX_TOKEN_NUM=1024 \
GRADIENT_ACCUMULATION_STEPS=16 \
OUTPUT_DIR=/root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft_debug \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

For multi-GPU ZeRO-2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
DEEPSPEED=zero2 \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

## Best Checkpoint Preservation

The launch script tracks the best validation checkpoint by default:

```bash
LOAD_BEST_MODEL_AT_END=true
METRIC_FOR_BEST_MODEL=eval_loss
GREATER_IS_BETTER=false
CREATE_CHECKPOINT_SYMLINK=true
PRESERVE_BEST_CHECKPOINT=true
```

`save_total_limit` still controls normal `checkpoint-*` rotation. The launcher
also starts `preserve_best_checkpoint.py` automatically and keeps an extra copy
of the best checkpoint under `<run-dir>/best_checkpoint`, outside Trainer's
`checkpoint-*` rotation.

To run the preservation script manually:

```bash
python examples/evdp/preserve_best_checkpoint.py \
  --run-dir /root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft/<run-dir> \
  --metric eval_loss \
  --mode min
```

For an active long run started without launcher auto-preservation, keep it
watching in the background:

```bash
setsid -f python examples/evdp/preserve_best_checkpoint.py \
  --run-dir /root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft/<run-dir> \
  --metric eval_loss \
  --mode min \
  --watch \
  --watch-interval 300 \
  > /root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft/<run-dir>/best_checkpoint_watcher.log 2>&1 &
```

This writes the protected copy to `<run-dir>/best_checkpoint`, which is not
matched by Trainer's `checkpoint-*` rotation.

## Resume After Interruption

The training script supports automatic checkpoint discovery:

```bash
cd /root/ms-swift
RESUME_FROM_CHECKPOINT=latest \
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
DATA_DIR=examples/evdp/data/mimic_cxr_report_sft \
ATTN_IMPL=sdpa \
PADDING_FREE=false \
PACKING=false \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

`latest` searches under `OUTPUT_DIR` for the newest `checkpoint-*`, sets
`--resume_from_checkpoint`, and resumes into the checkpoint's parent run
directory with `--add_version false`.

You can also resume from an explicit checkpoint:

```bash
RESUME_FROM_CHECKPOINT=/root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft/<run-dir>/checkpoint-500 \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

If the conda environment uses `torch<2.6`, Transformers may refuse to load
`optimizer.pt`/`scheduler.pt` because of the `torch.load` CVE guard. In that
case, resume model weights only:

```bash
RESUME_FROM_CHECKPOINT=/root/autodl-tmp/ms-swift/output/qwen3vl4b_mimic_report_sft/<run-dir>/checkpoint-5500 \
RESUME_ONLY_MODEL=true \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

This avoids loading optimizer/scheduler state. Use a `torch>=2.6` environment
if exact optimizer/scheduler resume is required.

## Flash Attention Notes

The default Qwen3-VL example uses:

```bash
ATTN_IMPL=flash_attn
PADDING_FREE=true
```

This requires a working `flash-attn` installation. On the current machine,
`torch==2.11.0+cu130` is installed, while the available CUDA compiler reports
CUDA 11.8, and no prebuilt flash-attn wheel was available for this exact
combination. Use the stable fallback unless you rebuild the environment:

```bash
ATTN_IMPL=sdpa \
PADDING_FREE=false \
PACKING=false \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

The launch script auto-falls back from `flash_attn` to `sdpa` when
`flash_attn` is not importable. To fail hard instead:

```bash
REQUIRE_FLASH_ATTN=1 \
ATTN_IMPL=flash_attn \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```
