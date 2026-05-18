#!/usr/bin/env python3
"""Copy the best eval checkpoint outside Trainer checkpoint rotation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="A concrete run directory, or an output root with --auto-latest-run.")
    parser.add_argument(
        "--auto-latest-run",
        action="store_true",
        help="If --run-dir has no logging.jsonl, use the newest child directory that has logging.jsonl.",
    )
    parser.add_argument("--metric", default="eval_loss")
    parser.add_argument("--mode", choices=["min", "max"], default="min")
    parser.add_argument("--dest", default=None, help="Default: <run-dir>/best_checkpoint")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=float, default=300.0)
    parser.add_argument("--stop-on-final", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_run_dir(path: Path, auto_latest_run: bool) -> Path | None:
    if (path / "logging.jsonl").exists():
        return path
    if not auto_latest_run or not path.exists():
        return None

    candidates = [p.parent for p in path.glob("*/logging.jsonl") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p / "logging.jsonl").stat().st_mtime)


def _step_from_record(record: dict[str, Any]) -> int | None:
    step_text = record.get("global_step/max_steps")
    if isinstance(step_text, str) and "/" in step_text:
        try:
            return int(step_text.split("/", 1)[0])
        except ValueError:
            return None
    step = record.get("global_step") or record.get("step")
    return int(step) if isinstance(step, int) else None


def find_best(logging_path: Path, metric: str, mode: str) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    if not logging_path.exists():
        return None

    with logging_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if metric not in record:
                continue
            step = _step_from_record(record)
            if step is None:
                continue
            value = float(record[metric])
            if best is None:
                best = (step, value)
            elif mode == "min" and value < best[1]:
                best = (step, value)
            elif mode == "max" and value > best[1]:
                best = (step, value)
    return best


def final_record_seen(logging_path: Path) -> bool:
    if not logging_path.exists():
        return False
    with logging_path.open("r", encoding="utf-8") as f:
        for line in f:
            if "last_model_checkpoint" in line:
                return True
    return False


def current_preserved_step(dest: Path) -> int | None:
    meta_path = dest / "best_checkpoint_meta.json"
    if not meta_path.exists():
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return int(meta["step"])
    except Exception:
        return None


def preserve_checkpoint(run_dir: Path, dest: Path, step: int, metric: str, value: float) -> bool:
    src = run_dir / f"checkpoint-{step}"
    if not src.is_dir():
        print(f"[preserve_best_checkpoint] Waiting for checkpoint directory: {src}", flush=True)
        return False
    if current_preserved_step(dest) == step:
        return True

    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(src, tmp, symlinks=True)

    meta = {
        "step": step,
        "metric": metric,
        "metric_value": value,
        "source_checkpoint": str(src),
        "preserved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (tmp / "best_checkpoint_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    if dest.exists():
        shutil.rmtree(dest)
    os.rename(tmp, dest)
    print(f"[preserve_best_checkpoint] Preserved checkpoint-{step} ({metric}={value}) -> {dest}", flush=True)
    return True


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_dir).resolve()

    while True:
        run_dir = resolve_run_dir(run_root, args.auto_latest_run)
        if run_dir is None:
            if not args.watch:
                raise SystemExit(1)
            print(f"[preserve_best_checkpoint] Waiting for run directory under: {run_root}", flush=True)
            time.sleep(args.watch_interval)
            continue

        logging_path = run_dir / "logging.jsonl"
        dest = Path(args.dest).resolve() if args.dest else run_dir / "best_checkpoint"
        best = find_best(logging_path, args.metric, args.mode)
        copied = False
        if best is not None:
            copied = preserve_checkpoint(run_dir, dest, best[0], args.metric, best[1])

        if not args.watch:
            raise SystemExit(0 if copied else 1)
        if args.stop_on_final and final_record_seen(logging_path) and copied:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
