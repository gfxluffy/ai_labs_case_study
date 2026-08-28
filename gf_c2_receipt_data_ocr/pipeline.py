"""
Orchestrates the OCR + entity-extraction stages into the four evaluation
conditions:

  raw_ocr             = basic OCR engine, no LLM post-processing
  raw_ocr_llm         = basic OCR engine + entity-analysis LLM   (Condition 3)
  multimodal_ocr      = vision LLM transcription, no post-processing (Condition 2)
  multimodal_ocr_llm  = vision LLM transcription + entity-analysis LLM (Condition 4)

Note: "raw_ocr" and "multimodal_ocr" without LLM post-processing don't
naturally produce the 4 structured fields (they're just blocks of text), so
for those two conditions this module applies a lightweight regex-based
field guesser instead of an LLM. This keeps the "no post-processing" arm of
the comparison honest -- it's not secretly using an LLM -- while still
letting you score it against ground truth on the same 4 fields.
"""

import re
from pathlib import Path

from entity_extraction import EntityExtractor
from ocr_basic import IsolatedBasicOCR
from ocr_multimodal import MultimodalOCR


def _naive_field_guess(raw_text: str) -> dict:
    """
    Rule-based fallback extraction for the "no LLM post-processing"
    conditions (Conditions 1 and 2). Deliberately simple -- the point of
    these conditions is to show what you get WITHOUT LLM-assisted
    structuring, so a strong heuristic here would defeat the comparison.
    """
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]

    company = lines[0] if lines else "unknown"

    date_match = re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", raw_text)
    date = date_match.group(0) if date_match else "unknown"

    total_match = re.search(
        r"(?:total|amount due|grand total)\D{0,10}(\d+[.,]\d{2})", raw_text, re.IGNORECASE
    )
    total = total_match.group(1) if total_match else "unknown"

    # crude address guess: first line containing a digit that isn't the date/total line
    address = "unknown"
    for line in lines[1:]:
        if any(ch.isdigit() for ch in line) and line not in (date, total):
            address = line
            break

    return {"company": company, "date": date, "address": address, "total": total}


class ReceiptPipeline:
    def __init__(self):
        self.basic_ocr = IsolatedBasicOCR()
        self.multimodal_ocr = MultimodalOCR()
        self.entity_extractor = EntityExtractor()

    def run_all_conditions(self, image_path: Path) -> dict:
        """
        Returns:
            {
                "raw_ocr":            {"raw_text": ..., "entities": {...}},
                "raw_ocr_llm":        {"raw_text": ..., "entities": {...}},
                "multimodal_ocr":     {"raw_text": ..., "entities": {...}},
                "multimodal_ocr_llm": {"raw_text": ..., "entities": {...}},
            }
        The two OCR stages are run once each and reused across their
        with-LLM / without-LLM variants, so each image only costs 1 basic
        OCR call + 1 vision LLM call + 2 entity-extraction LLM calls.

        Run sequentially rather than concurrently: on a CPU-only Ollama setup
        (no GPU) a single inference call already saturates all cores, so
        overlapping calls just contend for the same 4 cores and make each one
        slower -- measured ~5 min/receipt with contention vs. better without
        it, plus request timeouts under load. If you have a GPU-backed Ollama
        install (`ollama ps` shows GPU, not 100% CPU), concurrency is worth
        revisiting.
        """
        basic_text = self.basic_ocr.extract_text(image_path)
        multimodal_text = self.multimodal_ocr.extract_text(image_path)

        return {
            "raw_ocr": {
                "raw_text": basic_text,
                "entities": _naive_field_guess(basic_text),
            },
            "raw_ocr_llm": {
                "raw_text": basic_text,
                "entities": self.entity_extractor.extract_entities(basic_text),
            },
            "multimodal_ocr": {
                "raw_text": multimodal_text,
                "entities": _naive_field_guess(multimodal_text),
            },
            "multimodal_ocr_llm": {
                "raw_text": multimodal_text,
                "entities": self.entity_extractor.extract_entities(multimodal_text),
            },
        }

    def run_single_condition(self, image_path: Path, condition: str) -> dict:
        """Run just one condition -- used by the Gradio app so it doesn't pay
        for all four pipeline variants on every upload."""
        basic_needed = condition in ("raw_ocr", "raw_ocr_llm")
        multimodal_needed = condition in ("multimodal_ocr", "multimodal_ocr_llm")

        raw_text = (
            self.basic_ocr.extract_text(image_path)
            if basic_needed
            else self.multimodal_ocr.extract_text(image_path)
        )

        if condition in ("raw_ocr", "multimodal_ocr"):
            entities = _naive_field_guess(raw_text)
        elif condition in ("raw_ocr_llm", "multimodal_ocr_llm"):
            entities = self.entity_extractor.extract_entities(raw_text)
        else:
            raise ValueError(f"Unknown condition: {condition}")

        return {"raw_text": raw_text, "entities": entities}
