"""
Runs all four conditions across the SROIE V2 dataset and produces:
  - output/predictions_raw.csv         (every prediction, for auditing/debugging)
  - output/comparison_table.csv        (the headline 4-condition comparison)
  - output/errors.csv                  (receipts that failed, with reason + timing)
  - printed summary to stdout

No training happens anywhere in this pipeline (OCR engines and LLMs are all
frozen/pretrained), so there's no leakage risk in pooling train + test for a
larger evaluation sample -- see --pool below.

Usage:
    python run_evaluation.py                          # full test set only
    python run_evaluation.py --limit 20                # quick smoke test, first 20 receipts
    python run_evaluation.py --pool all --sample 200    # random 200-receipt sample from train+test combined
"""

import argparse
import csv
import random
import time
from pathlib import Path

import pandas as pd

import config
from evaluation import build_comparison_table, evaluate_condition
from pipeline import ReceiptPipeline
from sroie_dataset import list_receipts

CONDITIONS = ["raw_ocr", "raw_ocr_llm", "multimodal_ocr", "multimodal_ocr_llm"]

AUDIT_FIELDS = (
    ["image_id", "condition"]
    + [f"pred_{f}" for f in config.ENTITY_FIELDS]
    + [f"gt_{f}" for f in config.ENTITY_FIELDS]
)
ERROR_FIELDS = ["image_id", "error", "elapsed_seconds"]


class _CheckpointWriter:
    """Appends rows to a CSV as they're produced, so a long unattended run
    that gets interrupted (crash, sleep, power loss) still leaves partial
    results on disk instead of losing everything until the final write."""

    def __init__(self, path: Path, fieldnames: list[str], resume: bool = False):
        write_header = not (resume and path.exists())
        self._file = open(path, "a" if resume else "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict):
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        self._file.close()


def _load_completed(predictions_path: Path) -> tuple[dict, set]:
    """Reconstruct in-memory predictions from a previous (crashed/interrupted)
    run's checkpoint file, so resuming doesn't lose already-computed work.
    Only counts a receipt as done if all 4 conditions are present for it --
    a receipt that crashed partway through writing its 4 rows is treated as
    not done and gets reprocessed (its stale partial rows are harmless,
    since final scoring uses this reconstructed dict, not the raw CSV)."""
    predictions = {c: {} for c in CONDITIONS}
    if not predictions_path.exists():
        return predictions, set()

    df = pd.read_csv(predictions_path)
    if df.empty:
        return predictions, set()

    complete_ids = set()
    for image_id, group in df.groupby("image_id"):
        conditions_present = set(group["condition"])
        if conditions_present != set(CONDITIONS):
            continue
        complete_ids.add(image_id)
        for _, row in group.iterrows():
            predictions[row["condition"]][image_id] = {
                f: str(row[f"pred_{f}"]) for f in config.ENTITY_FIELDS
            }

    return predictions, complete_ids


def _load_pool(pool: str) -> list:
    if pool == "test":
        return list_receipts(config.TEST_IMG_DIR, config.TEST_ENTITIES_DIR)

    if pool == "all":
        test_receipts = list_receipts(config.TEST_IMG_DIR, config.TEST_ENTITIES_DIR)
        train_receipts = list_receipts(config.TRAIN_IMG_DIR, config.TRAIN_ENTITIES_DIR)

        seen_ids = {r.image_id for r in test_receipts}
        combined = list(test_receipts)
        dupes = 0
        for r in train_receipts:
            if r.image_id in seen_ids:
                dupes += 1
                continue
            seen_ids.add(r.image_id)
            combined.append(r)

        print(
            f"Pooled {len(test_receipts)} test + {len(train_receipts)} train "
            f"receipts ({dupes} duplicate image_ids skipped) = {len(combined)} total",
            flush=True,
        )
        return combined

    raise ValueError(f"Unknown pool: {pool}")


def main(
    limit: int | None = None,
    pool: str = "test",
    sample: int | None = None,
    seed: int = 42,
    resume: bool = False,
):
    receipts = _load_pool(pool)

    if sample:
        n = min(sample, len(receipts))
        # random.Random(seed).sample on the same pool + n is deterministic,
        # so a --resume run reproduces the exact same sample as the original.
        receipts = random.Random(seed).sample(receipts, n)
        print(f"Randomly sampled {n} receipts (seed={seed}) from the {pool} pool", flush=True)
    elif limit:
        receipts = receipts[:limit]

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    predictions = {c: {} for c in CONDITIONS}
    ground_truths = {}
    completed_ids = set()
    if resume:
        predictions, completed_ids = _load_completed(output_dir / "predictions_raw.csv")
        for receipt in receipts:
            if receipt.image_id in completed_ids:
                ground_truths[receipt.image_id] = receipt.ground_truth
        print(f"Resuming: {len(completed_ids)} receipts already completed, skipping them", flush=True)

    remaining_receipts = [r for r in receipts if r.image_id not in completed_ids]
    print(f"Evaluating on {len(remaining_receipts)} remaining receipts (of {len(receipts)} total)...", flush=True)

    pipeline = ReceiptPipeline()

    audit_rows = []
    error_rows = []
    durations = []

    audit_checkpoint = _CheckpointWriter(output_dir / "predictions_raw.csv", AUDIT_FIELDS, resume=resume)
    error_checkpoint = _CheckpointWriter(output_dir / "errors.csv", ERROR_FIELDS, resume=resume)

    run_start = time.monotonic()

    try:
        for i, receipt in enumerate(remaining_receipts, 1):
            receipt_start = time.monotonic()
            ground_truths[receipt.image_id] = receipt.ground_truth

            try:
                results = pipeline.run_all_conditions(receipt.image_path)
            except Exception as e:
                elapsed = time.monotonic() - receipt_start
                print(f"[{i}/{len(remaining_receipts)}] {receipt.image_id}  [ERROR after {elapsed:.0f}s] {e}", flush=True)
                error_row = {"image_id": receipt.image_id, "error": str(e), "elapsed_seconds": round(elapsed, 1)}
                error_rows.append(error_row)
                error_checkpoint.write(error_row)
                continue

            elapsed = time.monotonic() - receipt_start
            durations.append(elapsed)
            print(f"[{i}/{len(remaining_receipts)}] {receipt.image_id}  ({elapsed:.0f}s)", flush=True)

            for condition in CONDITIONS:
                predictions[condition][receipt.image_id] = results[condition]["entities"]
                audit_row = {
                    "image_id": receipt.image_id,
                    "condition": condition,
                    **{f"pred_{k}": v for k, v in results[condition]["entities"].items()},
                    **{f"gt_{k}": v for k, v in receipt.ground_truth.items()},
                }
                audit_rows.append(audit_row)
                audit_checkpoint.write(audit_row)
    finally:
        audit_checkpoint.close()
        error_checkpoint.close()

    total_elapsed = time.monotonic() - run_start

    condition_results = {
        c: evaluate_condition(predictions[c], ground_truths, c) for c in CONDITIONS
    }
    comparison_table = build_comparison_table(condition_results)

    # predictions_raw.csv and errors.csv were already written incrementally
    # via the checkpoint writers above; only the tables that need the full
    # in-memory results (comparison, per-field) are written here.
    comparison_table.to_csv(output_dir / "comparison_table.csv")

    per_field_dir = output_dir / "per_field"
    per_field_dir.mkdir(exist_ok=True)
    for condition, df in condition_results.items():
        df.to_csv(per_field_dir / f"{condition}.csv", index=False)

    n_ok_this_run = len(durations)
    n_ok = n_ok_this_run + len(completed_ids)
    n_err = len(error_rows)
    n_total = len(receipts)

    print("\n=== 4-Condition Comparison (OVERALL) ===", flush=True)
    print(comparison_table.to_string(), flush=True)

    print("\n=== Timing ===", flush=True)
    print(f"This run's wall time: {total_elapsed / 60:.1f} min ({total_elapsed:.0f}s)", flush=True)
    if completed_ids:
        print(f"({len(completed_ids)} receipts were already done from a prior run and resumed, not re-timed)", flush=True)
    print(f"Receipts: {n_ok} succeeded, {n_err} errored this run (of {n_total} total in the sample)", flush=True)
    if durations:
        print(
            f"Per-receipt this run (succeeded only): avg {sum(durations) / len(durations):.1f}s, "
            f"min {min(durations):.1f}s, max {max(durations):.1f}s",
            flush=True,
        )
    if n_total:
        error_rate = n_err / n_total
        print(f"Error rate: {error_rate:.1%}", flush=True)
        if error_rate > 0.10:
            print(
                "WARNING: error rate above 10% -- check output/errors.csv before trusting "
                "the comparison table above.",
                flush=True,
            )

    print(f"\nFull results written to {output_dir}/", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on only the first N receipts (quick smoke test)")
    parser.add_argument("--pool", choices=["test", "all"], default="test", help="'test' (default, original behavior) or 'all' (train+test combined -- safe since nothing is trained on this data)")
    parser.add_argument("--sample", type=int, default=None, help="Randomly sample N receipts from the pool (takes priority over --limit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample, for a reproducible subsample")
    parser.add_argument("--resume", action="store_true", help="Skip receipts already completed in output/predictions_raw.csv from a prior (e.g. crashed) run of the same --pool/--sample/--seed")
    args = parser.parse_args()
    main(limit=args.limit, pool=args.pool, sample=args.sample, seed=args.seed, resume=args.resume)
