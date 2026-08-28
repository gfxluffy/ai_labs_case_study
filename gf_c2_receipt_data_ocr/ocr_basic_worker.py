"""
Subprocess entry point for running BasicOCR.extract_text on a single image.

Invoked by ocr_basic.IsolatedBasicOCR, not run directly. Exists because
EasyOCR's CPU/quantized backend has been observed to segfault deterministically
on certain receipt images (e.g. unusual aspect ratios) -- a native crash that
no try/except in the parent process can catch. Running each call in its own
subprocess means a crash here only fails that one receipt; the parent detects
it via a non-zero/missing-output exit and logs a normal, catchable error.

Usage: python ocr_basic_worker.py <image_path> <output_text_path>
"""

import sys
from pathlib import Path

from ocr_basic import BasicOCR


def main():
    image_path, output_path = sys.argv[1], sys.argv[2]
    ocr = BasicOCR()
    text = ocr.extract_text(image_path)
    Path(output_path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
