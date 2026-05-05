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
OUTPUT_DIR=output/qwen3vl4b_mimic_report_sft_debug \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```

For multi-GPU ZeRO-2:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
DEEPSPEED=zero2 \
bash examples/evdp/run_qwen3vl4b_report_sft.sh
```
