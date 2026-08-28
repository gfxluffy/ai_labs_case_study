"""
Evaluation harness.

Scores a pipeline condition's predictions against SROIE ground truth on:
  - "number of entities extracted": did the pipeline attempt each field at
    all (i.e. not "unknown"/empty)?
  - "correctness of extracted entities": exact-match rate and fuzzy
    similarity per field, since OCR-derived text rarely matches ground
    truth character-for-character even when it's substantively correct.

Produces both a per-field breakdown and a single aggregate score per
condition, so you can build the 4-condition comparison table the assignment
asks for.
"""

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import pandas as pd

from config import ENTITY_FIELDS
from sroie_dataset import normalize, normalize_amount


def _similarity(a: str, b: str) -> float:
    """Character-level similarity ratio in [0, 1] using difflib (no extra deps)."""
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def _field_normalizer(field_name: str):
    return normalize_amount if field_name == "total" else normalize


@dataclass
class FieldScore:
    field: str
    attempted: int = 0       # pipeline returned something other than "unknown"/empty
    total: int = 0           # number of receipts evaluated
    exact_matches: int = 0
    similarity_sum: float = 0.0

    @property
    def extraction_rate(self) -> float:
        return self.attempted / self.total if self.total else 0.0

    @property
    def exact_match_rate(self) -> float:
        return self.exact_matches / self.total if self.total else 0.0

    @property
    def avg_similarity(self) -> float:
        return self.similarity_sum / self.total if self.total else 0.0


def evaluate_condition(
    predictions: dict[str, dict],
    ground_truths: dict[str, dict],
    condition_name: str,
) -> pd.DataFrame:
    """
    predictions / ground_truths: {image_id: {"company": ..., "date": ..., ...}}
    Both dicts must be keyed by the same image_ids for a fair comparison;
    image_ids present in ground_truths but missing from predictions are
    treated as a total failure to extract (counts against extraction_rate).

    Returns a DataFrame with one row per field plus a final "overall" row,
    all tagged with the condition name so results from multiple conditions
    can be concatenated into one comparison table.
    """
    scores = {f: FieldScore(field=f) for f in ENTITY_FIELDS}

    for image_id, gt in ground_truths.items():
        pred = predictions.get(image_id, {f: "" for f in ENTITY_FIELDS})

        for f in ENTITY_FIELDS:
            normalizer = _field_normalizer(f)
            pred_val = normalizer(pred.get(f, ""))
            gt_val = normalizer(gt.get(f, ""))

            scores[f].total += 1
            if pred_val and pred_val != "unknown":
                scores[f].attempted += 1
            if pred_val == gt_val and gt_val != "":
                scores[f].exact_matches += 1
            scores[f].similarity_sum += _similarity(pred_val, gt_val)

    rows = []
    for f in ENTITY_FIELDS:
        s = scores[f]
        rows.append(
            {
                "condition": condition_name,
                "field": f,
                "n_receipts": s.total,
                "extraction_rate": round(s.extraction_rate, 3),
                "exact_match_rate": round(s.exact_match_rate, 3),
                "avg_similarity": round(s.avg_similarity, 3),
            }
        )

    # Overall row = simple average across fields, for a quick one-number
    # comparison in the results slide.
    overall = {
        "condition": condition_name,
        "field": "OVERALL",
        "n_receipts": scores[ENTITY_FIELDS[0]].total,
        "extraction_rate": round(sum(r["extraction_rate"] for r in rows) / len(rows), 3),
        "exact_match_rate": round(sum(r["exact_match_rate"] for r in rows) / len(rows), 3),
        "avg_similarity": round(sum(r["avg_similarity"] for r in rows) / len(rows), 3),
    }
    rows.append(overall)

    return pd.DataFrame(rows)


def build_comparison_table(condition_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    condition_results: {condition_name: DataFrame from evaluate_condition(...)}
    Concatenates all conditions and pivots to a clean side-by-side table,
    OVERALL row only, for the headline slide comparing all four conditions.
    """
    combined = pd.concat(condition_results.values(), ignore_index=True)
    overall_only = combined[combined["field"] == "OVERALL"].drop(columns=["field", "n_receipts"])
    return overall_only.set_index("condition")
