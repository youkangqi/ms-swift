#!/usr/bin/env python3
"""Build report-level MIMIC-CXR JSONL files for ms-swift multimodal SFT.

The output follows the ms-swift standard multimodal SFT format:

{"messages": [{"role": "system", "content": "..."}, {"role": "user",
"content": "<image>..."}, {"role": "assistant", "content": "..."}],
"images": ["/abs/path/to/image.jpg"]}

Samples are study-level: all selected images from one radiology study are
provided as input and the corresponding report is used as the target.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_SYSTEM_PROMPT = (
    "You are a radiology report generation assistant. Generate a concise chest X-ray "
    "radiology report based only on the provided image(s)."
)

DEFAULT_USER_PROMPT = (
    "Generate the chest X-ray radiology report for the provided image(s). "
    "Include Findings and Impression when supported by the image(s)."
)

VIEW_PRIORITY = [
    "PA",
    "AP",
    "AP AXIAL",
    "AP LLD",
    "AP RLD",
    "LATERAL",
    "LL",
    "LAO",
    "RAO",
]

SECTION_TITLE = {
    "findings": "Findings",
    "impression": "Impression",
    "findings_impression": "Findings and Impression",
}

ADMIN_SECTIONS = {
    "examination",
    "indication",
    "technique",
    "comparison",
    "history",
    "reason for exam",
    "reason for examination",
}

HEADING_RE = re.compile(r"^\s*([A-Z][A-Z0-9 /(),._-]{1,100}):\s*(.*)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mimic-root",
        type=Path,
        default=Path("/root/autodl-tmp/mimic-cxr-jpg/2.1.0"),
        help="MIMIC-CXR-JPG 2.1.0 root directory.",
    )
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Path to mimic-cxr metadata CSV.")
    parser.add_argument("--split-csv", type=Path, default=None, help="Path to mimic-cxr split CSV or CSV.GZ.")
    parser.add_argument("--image-root", type=Path, default=None, help="Root containing JPG files.")
    parser.add_argument("--report-root", type=Path, default=None, help="Root containing report txt files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/ms-swift/examples/evdp/data/mimic_cxr_report_sft"),
        help="Directory to write train/val/test JSONL and summary files.",
    )
    parser.add_argument(
        "--max-images-per-study",
        type=int,
        default=2,
        help="Maximum selected image files per study. Set <=0 to keep all images.",
    )
    parser.add_argument(
        "--view-positions",
        nargs="*",
        default=None,
        help="Optional view-position allowlist, e.g. PA AP LATERAL. Default keeps all views.",
    )
    parser.add_argument(
        "--target-sections",
        nargs="+",
        choices=["findings", "impression"],
        default=["findings", "impression"],
        help="Report sections to use as the assistant target.",
    )
    parser.add_argument(
        "--no-fallback-full-report",
        action="store_true",
        help="Skip reports where target sections cannot be extracted instead of falling back to cleaned full text.",
    )
    parser.add_argument("--min-report-chars", type=int, default=20, help="Skip target reports shorter than this.")
    parser.add_argument("--max-report-chars", type=int, default=6000, help="Skip target reports longer than this.")
    parser.add_argument("--max-samples-per-split", type=int, default=None, help="Debug cap applied to every split.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Debug cap for train split.")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Debug cap for val split.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Debug cap for test split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used before applying sample caps.")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", default=DEFAULT_USER_PROMPT)
    parser.add_argument(
        "--include-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include trace metadata fields in each JSONL row.",
    )
    return parser.parse_args()


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def normalize_split(split: str) -> str:
    split = (split or "train").strip().lower()
    if split in {"valid", "validate", "validation", "val"}:
        return "val"
    if split in {"test"}:
        return "test"
    return "train"


def subject_prefix(subject_id: str) -> str:
    subject_id = str(subject_id).strip()
    return f"p{subject_id[:2]}"


def report_path(report_root: Path, subject_id: str, study_id: str) -> Path:
    subject_id = str(subject_id).strip()
    study_id = str(study_id).strip()
    return report_root / subject_prefix(subject_id) / f"p{subject_id}" / f"s{study_id}.txt"


def image_path(image_root: Path, subject_id: str, study_id: str, dicom_id: str) -> Path:
    subject_id = str(subject_id).strip()
    study_id = str(study_id).strip()
    dicom_id = str(dicom_id).strip()
    return image_root / subject_prefix(subject_id) / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def read_split_map(split_csv: Path) -> Dict[str, str]:
    split_by_dicom: Dict[str, str] = {}
    with open_text(split_csv) as f:
        for row in csv.DictReader(f):
            dicom_id = row.get("dicom_id")
            if not dicom_id:
                continue
            split_by_dicom[dicom_id] = normalize_split(row.get("split", "train"))
    return split_by_dicom


def clean_heading(raw_heading: str) -> str:
    heading = raw_heading.strip().lower()
    heading = re.sub(r"[_\W]+", " ", heading)
    return re.sub(r"\s+", " ", heading).strip()


def canonical_section(raw_heading: str) -> Optional[str]:
    heading = clean_heading(raw_heading)
    if not heading:
        return None
    if "finding" in heading and "impression" in heading:
        return "findings_impression"
    if heading.startswith("finding"):
        return "findings"
    if heading.startswith("impression"):
        return "impression"
    if heading in ADMIN_SECTIONS:
        return heading
    return heading


def collapse_blank_lines(lines: Iterable[str]) -> List[str]:
    out: List[str] = []
    previous_blank = True
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line.strip())
        if not line:
            if not previous_blank:
                out.append("")
            previous_blank = True
            continue
        out.append(line)
        previous_blank = False
    while out and out[-1] == "":
        out.pop()
    return out


def parse_sections(raw_text: str) -> List[Tuple[str, List[str]]]:
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    sections: List[Tuple[str, List[str]]] = []
    current_heading: Optional[str] = None
    current_lines: List[str] = []

    def flush() -> None:
        nonlocal current_heading, current_lines
        if current_heading is not None:
            cleaned = collapse_blank_lines(current_lines)
            if cleaned:
                sections.append((current_heading, cleaned))
        current_heading = None
        current_lines = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            current_lines.append("")
            continue
        if stripped == "FINAL REPORT":
            continue
        match = HEADING_RE.match(stripped)
        if match:
            heading = canonical_section(match.group(1))
            if heading:
                flush()
                current_heading = heading
                inline_content = match.group(2).strip()
                current_lines = [inline_content] if inline_content else []
                continue
        if current_heading is None:
            current_heading = "full_report"
        current_lines.append(stripped)
    flush()
    return sections


def extract_target_report(raw_text: str, target_sections: Sequence[str], fallback_full_report: bool) -> str:
    sections = parse_sections(raw_text)
    wanted = set(target_sections)
    selected: List[str] = []
    used_combined = False

    for heading, lines in sections:
        if heading == "findings_impression" and {"findings", "impression"} & wanted:
            selected.extend([f"{SECTION_TITLE[heading]}:", *lines, ""])
            used_combined = True
        elif heading in wanted:
            selected.extend([f"{SECTION_TITLE[heading]}:", *lines, ""])

    if selected:
        return "\n".join(collapse_blank_lines(selected)).strip()

    if not fallback_full_report:
        return ""

    fallback_lines: List[str] = []
    for heading, lines in sections:
        if heading in ADMIN_SECTIONS:
            continue
        if used_combined and heading in wanted:
            continue
        if heading in SECTION_TITLE:
            fallback_lines.extend([f"{SECTION_TITLE[heading]}:", *lines, ""])
        elif heading == "full_report":
            fallback_lines.extend([*lines, ""])
        else:
            fallback_lines.extend(lines + [""])
    return "\n".join(collapse_blank_lines(fallback_lines)).strip()


def view_rank(view_position: str) -> Tuple[int, str]:
    view = (view_position or "").strip().upper()
    try:
        return (VIEW_PRIORITY.index(view), view)
    except ValueError:
        return (len(VIEW_PRIORITY), view)


def select_images(rows: List[dict], image_root: Path, max_images_per_study: int) -> Tuple[List[str], List[str], List[str]]:
    sorted_rows = sorted(rows, key=lambda r: (view_rank(r.get("ViewPosition", "")), r.get("dicom_id", "")))
    if max_images_per_study > 0:
        sorted_rows = sorted_rows[:max_images_per_study]

    images: List[str] = []
    dicom_ids: List[str] = []
    views: List[str] = []
    for row in sorted_rows:
        path = image_path(image_root, row["subject_id"], row["study_id"], row["dicom_id"])
        if not path.exists():
            continue
        images.append(str(path.resolve()))
        dicom_ids.append(row["dicom_id"])
        views.append((row.get("ViewPosition") or "").strip())
    return images, dicom_ids, views


def load_study_rows(
    metadata_csv: Path,
    split_by_dicom: Dict[str, str],
    allowed_views: Optional[set],
) -> Dict[Tuple[str, str, str], List[dict]]:
    grouped: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    with open_text(metadata_csv) as f:
        for row in csv.DictReader(f):
            dicom_id = row.get("dicom_id")
            subject_id = row.get("subject_id")
            study_id = row.get("study_id")
            if not dicom_id or not subject_id or not study_id:
                continue
            view = (row.get("ViewPosition") or "").strip().upper()
            if allowed_views is not None and view not in allowed_views:
                continue
            split = split_by_dicom.get(dicom_id, "train")
            key = (normalize_split(split), str(subject_id).strip(), str(study_id).strip())
            grouped[key].append(row)
    return grouped


def build_messages(system_prompt: str, user_prompt: str, num_images: int, report: str) -> List[dict]:
    image_tokens = "".join("<image>" for _ in range(num_images))
    user_content = f"{image_tokens}\n{user_prompt}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": report},
    ]


def split_cap(args: argparse.Namespace, split: str) -> Optional[int]:
    if split == "train" and args.max_train_samples is not None:
        return args.max_train_samples
    if split == "val" and args.max_val_samples is not None:
        return args.max_val_samples
    if split == "test" and args.max_test_samples is not None:
        return args.max_test_samples
    return args.max_samples_per_split


def build_item(
    split: str,
    subject_id: str,
    study_id: str,
    rows: List[dict],
    args: argparse.Namespace,
    image_root: Path,
    report_root: Path,
    fallback_full_report: bool,
    stats: Counter,
) -> Optional[dict]:
    path = report_path(report_root, subject_id, study_id)
    if not path.exists():
        stats["missing_report"] += 1
        return None
    images, dicom_ids, views = select_images(rows, image_root, args.max_images_per_study)
    if not images:
        stats["missing_images"] += 1
        return None
    raw_report = path.read_text(encoding="utf-8", errors="replace")
    report = extract_target_report(raw_report, args.target_sections, fallback_full_report)
    if len(report) < args.min_report_chars:
        stats["short_report"] += 1
        return None
    if args.max_report_chars and len(report) > args.max_report_chars:
        stats["long_report"] += 1
        return None

    item = {
        "messages": build_messages(args.system_prompt, args.user_prompt, len(images), report),
        "images": images,
    }
    if args.include_metadata:
        item.update({
            "id": f"p{subject_id}_s{study_id}",
            "split": split,
            "subject_id": subject_id,
            "study_id": study_id,
            "dicom_ids": dicom_ids,
            "view_positions": views,
            "report_path": str(path.resolve()),
        })
    return item


def write_jsonl(path: Path, rows: List[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    mimic_root = args.mimic_root.resolve()
    metadata_csv = args.metadata_csv or mimic_root / "mimic-cxr-2.0.0-metadata.csv"
    split_csv = args.split_csv or mimic_root / "mimic-cxr-2.0.0-split.csv.gz"
    image_root = args.image_root or mimic_root / "files"
    report_root = args.report_root or mimic_root / "report_files" / "files"
    output_dir = args.output_dir.resolve()

    for required in [metadata_csv, split_csv, image_root, report_root]:
        if not required.exists():
            raise FileNotFoundError(f"Required path not found: {required}")

    allowed_views = None
    if args.view_positions:
        allowed_views = {view.upper() for view in args.view_positions}

    random.seed(args.seed)
    split_by_dicom = read_split_map(split_csv)
    grouped = load_study_rows(metadata_csv, split_by_dicom, allowed_views)

    rows_by_split: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    stats = Counter()
    fallback_full_report = not args.no_fallback_full_report

    grouped_keys_by_split: Dict[str, List[Tuple[str, str, str]]] = {"train": [], "val": [], "test": []}
    for key in grouped:
        split, _, _ = key
        if split not in rows_by_split:
            continue
        grouped_keys_by_split[split].append(key)

    for split, keys in grouped_keys_by_split.items():
        random.shuffle(keys)
        cap = split_cap(args, split)
        stats[f"{split}_candidate_studies"] = len(keys)
        for key in keys:
            if cap is not None and len(rows_by_split[split]) >= cap:
                break
            _, subject_id, study_id = key
            stats["candidate_studies"] += 1
            item = build_item(
                split,
                subject_id,
                study_id,
                grouped[key],
                args,
                image_root,
                report_root,
                fallback_full_report,
                stats,
            )
            if item is None:
                continue
            rows_by_split[split].append(item)
            stats[f"{split}_rows_before_cap"] += 1

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mimic_root": str(mimic_root),
        "metadata_csv": str(metadata_csv),
        "split_csv": str(split_csv),
        "image_root": str(image_root),
        "report_root": str(report_root),
        "output_dir": str(output_dir),
        "max_images_per_study": args.max_images_per_study,
        "view_positions": sorted(allowed_views) if allowed_views else None,
        "target_sections": args.target_sections,
        "fallback_full_report": fallback_full_report,
        "splits": {},
        "skips": {
            "missing_report": stats["missing_report"],
            "missing_images": stats["missing_images"],
            "short_report": stats["short_report"],
            "long_report": stats["long_report"],
        },
    }

    for split, rows in rows_by_split.items():
        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        summary["splits"][split] = {
            "path": str(out_path),
            "num_rows": len(rows),
            "rows_before_cap": stats[f"{split}_rows_before_cap"],
            "candidate_studies": stats[f"{split}_candidate_studies"],
        }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
