"""
"Improved OCR" stage -- a multimodal (vision) LLM reads the receipt image
directly and transcribes it, instead of running a traditional OCR engine.

This produces the raw text used for Condition 2 (Improved OCR) and, after
entity extraction, Condition 4 (Improved OCR + entity-analysis LLM).

Uses a local Ollama vision model (e.g. llava, qwen2.5vl, moondream) so there
is no per-call API cost. If you'd rather use a hosted multimodal API
(OpenAI/Anthropic/Gemini vision), swap the internals of `extract_text` --
the rest of the pipeline only depends on this function's signature.
"""

import base64
import io
from pathlib import Path

import requests
from PIL import Image

import config

# Downscale before sending to the vision model -- receipt text stays legible
# well below full camera/scan resolution, and fewer pixels means fewer
# vision tokens for the model to process, which matters a lot on CPU
# inference. Only shrinks images larger than this; never upscales.
MAX_IMAGE_DIMENSION = 1024

TRANSCRIBE_PROMPT = (
    "Transcribe ALL text visible on this receipt image exactly as it appears, "
    "line by line, top to bottom. Include the store/company name, address, "
    "date, all line items, and the total amount. Do not summarize, explain, "
    "or add commentary -- output only the transcribed text."
)


class MultimodalOCR:
    def __init__(self, model: str = None, base_url: str = None):
        self.model = model or config.VISION_MODEL
        self.base_url = base_url or config.OLLAMA_BASE_URL

    def extract_text(self, image_path: Path) -> str:
        image_b64 = self._encode_image(image_path)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": TRANSCRIBE_PROMPT,
                "images": [image_b64],
                "stream": False,
                "keep_alive": config.OLLAMA_KEEP_ALIVE,
            },
            timeout=config.REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    @staticmethod
    def _encode_image(image_path: Path) -> str:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            if max(img.size) > MAX_IMAGE_DIMENSION:
                img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
