"""
"Basic OCR" stage -- a traditional (non-LLM) OCR engine.

This produces the raw text used for Condition 1 (Raw OCR) and, after entity
extraction, Condition 3 (Raw OCR + entity-analysis LLM).

Three backends are supported so you can pick whichever is easiest to install
in your environment; EasyOCR is the default since it needs no external
system binary. Swap BASIC_OCR_BACKEND in config.py to change engines --
nothing else in the pipeline needs to change.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import config

# Some SROIE receipts are ~5000x7000px camera photos. EasyOCR's CRAFT
# detector allocates feature maps proportional to input size, and on this
# CPU-only, memory-constrained setup that blew past available RAM (observed:
# "not enough memory: you tried to allocate 1.1-1.2 GB" on a single conv2d)
# -- which surfaced as an uncatchable native crash rather than a clean
# Python exception under torch's quantized CPU backend. Downscaling first
# keeps memory use bounded; 2000px preserves receipt text legibility while
# cutting the largest images to a small fraction of their original area.
MAX_BASIC_OCR_DIMENSION = 2000


class IsolatedBasicOCR:
    """Runs BasicOCR in a subprocess per call, so a native crash in the
    underlying engine (observed: EasyOCR's CPU/quantized backend segfaults
    deterministically on some receipt images -- see output/errors.csv)
    only fails that one receipt instead of killing the whole evaluation run.
    Costs a fresh model load per call (a few seconds) in exchange for that
    isolation; worth it since the alternative is losing the entire run to
    one poison-pill image with no way to catch it in-process."""

    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds
        self._worker_script = Path(__file__).parent / "ocr_basic_worker.py"

    def extract_text(self, image_path: Path) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "ocr_output.txt"
            try:
                result = subprocess.run(
                    [sys.executable, str(self._worker_script), str(image_path), str(output_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"Basic OCR subprocess timed out on {image_path}") from e

            if result.returncode != 0 or not output_path.exists():
                stderr_tail = result.stderr[-500:] if result.stderr else "(no stderr)"
                raise RuntimeError(
                    f"Basic OCR subprocess failed (returncode={result.returncode}) "
                    f"on {image_path}: {stderr_tail}"
                )
            return output_path.read_text(encoding="utf-8")


class BasicOCR:
    def __init__(self, backend: str = None, languages: list[str] = None):
        self.backend = backend or config.BASIC_OCR_BACKEND
        self.languages = languages or config.OCR_LANGUAGES
        self._engine = None
        self._load_engine()

    def _load_engine(self):
        if self.backend == "easyocr":
            import easyocr  # pip install easyocr

            # quantize=True (default): uses int8 weights, meaningfully lower
            # memory footprint than float32 -- matters on this machine,
            # which has been observed running with as little as ~1.8GB free
            # RAM (Ollama alone holds ~7GB resident). Whether an OOM here
            # surfaces as a native crash or a clean exception doesn't matter
            # for reliability: IsolatedBasicOCR already isolates this call in
            # a subprocess and catches failure via return code either way.
            self._engine = easyocr.Reader(self.languages, gpu=config.BASIC_OCR_GPU, quantize=True)

        elif self.backend == "paddleocr":
            from paddleocr import PaddleOCR  # pip install paddleocr paddlepaddle

            self._engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

        elif self.backend == "tesseract":
            import pytesseract  # pip install pytesseract; requires the tesseract binary

            self._engine = pytesseract

        else:
            raise ValueError(f"Unknown OCR backend: {self.backend}")

    def extract_text(self, image_path: Path) -> str:
        """Run OCR on a single receipt image and return the raw extracted text."""
        image_path = str(image_path)

        if self.backend == "easyocr":
            import numpy as np

            img = self._load_capped(image_path)
            results = self._engine.readtext(np.array(img), detail=0)
            return "\n".join(results)

        if self.backend == "paddleocr":
            results = self._engine.ocr(image_path, cls=True)
            lines = []
            for block in results:
                for line in block:
                    lines.append(line[1][0])
            return "\n".join(lines)

        if self.backend == "tesseract":
            return self._engine.image_to_string(self._load_capped(image_path))

        raise ValueError(f"Unknown OCR backend: {self.backend}")

    @staticmethod
    def _load_capped(image_path: str, max_dim: int = MAX_BASIC_OCR_DIMENSION):
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return img
