"""
"Entity-analysis LLM" stage -- takes raw OCR text (from either the basic
engine or the multimodal engine) and:
  1. structures it into the four SROIE fields (company, date, address, total)
  2. rectifies malformed entities caused by OCR noise (e.g. "sollor" -> "seller")
  3. completes compound entities with missing components (e.g. truncated
     vendor names)

This is what turns Condition 1 -> Condition 3, and Condition 2 -> Condition 4.
"""

import json
import re

import requests

import config

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# Design notes (worth mentioning in your presentation's methodology slide):
#  - Few-shot style instruction with an explicit output schema keeps the LLM
#    from returning prose or partial JSON.
#  - The "rectification" and "completion" instructions are called out
#    explicitly and separately, mirroring the two correction behaviors the
#    assignment asks you to demonstrate and document.
#  - "unknown" as the fallback value (rather than empty string or null) makes
#    it unambiguous during evaluation whether the model attempted extraction.
ENTITY_EXTRACTION_PROMPT = """You are an information-extraction assistant for retail receipts.

You will be given raw OCR text extracted from a scanned receipt. The OCR text
may contain errors: misrecognized characters, merged/split words, missing
characters, or garbled entities (for example "sollor" instead of "seller",
or a company name that is cut off mid-word).

Your job:
1. Extract exactly these four fields: company, date, address, total.
2. RECTIFY malformed entities: if a word is clearly a garbled OCR version of
   a real word or name, correct it to the most plausible real-world reading.
3. COMPLETE compound entities that are missing components: if a company name,
   address, or date is partially cut off but the missing part can be
   reasonably inferred from context, complete it.
4. If a field genuinely cannot be determined from the text, use "unknown" --
   do not guess wildly or invent values that aren't supported by the text.
5. Normalize the date to DD/MM/YYYY if a date is present in any recognizable format.
6. Normalize the total to a plain number with two decimal places (no currency symbol).

Respond with ONLY a JSON object in exactly this shape, no other text:
{{
  "company": "...",
  "date": "...",
  "address": "...",
  "total": "..."
}}

OCR TEXT:
---
{ocr_text}
---
"""


class EntityExtractor:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.TEXT_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL

    def extract_entities(self, raw_ocr_text: str) -> dict:
        """
        Given raw OCR text, return a dict with the four SROIE fields.
        Falls back to a dict of "unknown" values if the LLM response can't
        be parsed, so downstream evaluation code never has to special-case
        a missing result.
        """
        prompt = ENTITY_EXTRACTION_PROMPT.format(ocr_text=raw_ocr_text)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",  # ask Ollama to constrain output to valid JSON
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
                # The output is always a small 4-field JSON object -- capping
                # generation length prevents a rambling model from wasting
                # CPU time past the point where the answer is already done.
                "options": {"num_predict": 200},
            },
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        raw_response = response.json().get("response", "")

        return self._parse_response(raw_response)

    @staticmethod
    def _parse_response(raw_response: str) -> dict:
        from config import ENTITY_FIELDS

        fallback = {field: "unknown" for field in ENTITY_FIELDS}

        # Strip markdown code fences if the model added them despite instructions.
        cleaned = re.sub(r"^```(?:json)?|```$", "", raw_response.strip(), flags=re.MULTILINE).strip()

        # If there's leading/trailing prose, pull out the first {...} block.
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return fallback

        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return fallback

        return {field: str(parsed.get(field, "unknown")) or "unknown" for field in ENTITY_FIELDS}
