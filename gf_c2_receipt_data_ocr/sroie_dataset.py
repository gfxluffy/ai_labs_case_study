"""
Loading and normalizing SROIE V2 data.

SROIE V2's entity ground truth files are plain-text files containing a JSON
object with the four key fields. This module handles reading those, matching
them to their corresponding images, and normalizing strings so that later
comparison (OCR output vs. ground truth) isn't tripped up by whitespace,
case, or punctuation differences that don't reflect real extraction errors.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Receipt:
    image_id: str
    image_path: Path
    ground_truth: dict  # {"company": ..., "date": ..., "address": ..., "total": ...}


def list_receipts(img_dir: Path, entities_dir: Path) -> list[Receipt]:
    """
    Pair up every image in img_dir with its ground-truth entities file
    (same stem, .txt extension) in entities_dir. Images without a matching
    ground-truth file are skipped with a warning, since they can't be used
    for evaluation (though they'd still work fine through the app/pipeline).
    """
    img_dir = Path(img_dir)
    entities_dir = Path(entities_dir)

    if not img_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {img_dir}. Check config.py paths."
        )

    receipts = []
    image_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_paths:
        gt_path = entities_dir / f"{img_path.stem}.txt"
        if not gt_path.exists():
            print(f"[warn] no ground truth for {img_path.name}, skipping")
            continue

        gt = load_ground_truth(gt_path)
        if gt is None:
            print(f"[warn] could not parse ground truth for {img_path.name}, skipping")
            continue

        receipts.append(Receipt(image_id=img_path.stem, image_path=img_path, ground_truth=gt))

    return receipts


def load_ground_truth(gt_path: Path) -> Optional[dict]:
    """Parse a single SROIE entities .txt file into a dict of the 4 fields."""
    try:
        raw = Path(gt_path).read_text(encoding="utf-8", errors="ignore")
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return None

    # Ensure all expected fields exist, defaulting to empty string if missing.
    from config import ENTITY_FIELDS

    return {field: str(data.get(field, "")) for field in ENTITY_FIELDS}


def normalize(text: Optional[str]) -> str:
    """
    Normalize a string for fair comparison:
    - lowercase
    - collapse repeated whitespace
    - strip leading/trailing whitespace
    - remove punctuation that OCR commonly gets wrong/inconsistent on
      (commas, periods used as thousand/decimal separators are kept for
      totals -- see normalize_amount below for numeric fields)
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,-]", "", text)
    return text.strip()


def normalize_amount(text: Optional[str]) -> str:
    """
    Normalize a monetary total for comparison: strip currency symbols,
    thousand separators, and trailing zeros noise, keep only digits and one
    decimal point.
    """
    if not text:
        return ""
    cleaned = re.sub(r"[^\d.]", "", text)
    # Collapse multiple dots (e.g. "12.50.00" artifacts) to the first one.
    parts = cleaned.split(".")
    if len(parts) > 2:
        cleaned = parts[0] + "." + "".join(parts[1:])
    try:
        return f"{float(cleaned):.2f}" if cleaned else ""
    except ValueError:
        return cleaned
