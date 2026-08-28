"""
Central configuration for the receipt extraction pipeline.

Adjust the paths below to match where you've unzipped the SROIE V2 dataset.
Expected SROIE V2 layout (standard release):

    data/sroie_v2/
        train/
            img/            *.jpg receipt images
            entities/       *.txt files, each a JSON dict:
                             {"company": ..., "date": ..., "address": ..., "total": ...}
        test/
            img/
            entities/

If your copy of the dataset uses different subfolder names, just edit the
paths below -- nothing elsewhere in the codebase hardcodes them.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------
DATASET_ROOT = Path("./data/sroie_v2")

TRAIN_IMG_DIR = DATASET_ROOT / "train" / "img"
TRAIN_ENTITIES_DIR = DATASET_ROOT / "train" / "entities"

TEST_IMG_DIR = DATASET_ROOT / "test" / "img"
TEST_ENTITIES_DIR = DATASET_ROOT / "test" / "entities"

ENTITY_FIELDS = ["company", "date", "address", "total"]

# ---------------------------------------------------------------------------
# Basic OCR engine choice: "easyocr" | "paddleocr" | "tesseract"
# EasyOCR is the default because it's pure-Python (pip installable) with no
# external binary dependency, which keeps this runnable on a local machine
# without extra system setup.
# ---------------------------------------------------------------------------
BASIC_OCR_BACKEND = "easyocr"
OCR_LANGUAGES = ["en"]

# Set True if you have a working CUDA GPU + GPU-enabled torch install -- gives
# a large speedup on the basic-OCR stage. Defaults to False since a GPU
# install isn't guaranteed on every machine and a bad CUDA setup will error
# instead of silently falling back.
BASIC_OCR_GPU = False

# ---------------------------------------------------------------------------
# Local LLM settings (via Ollama -- free, runs on your own machine, no API
# cost). Install Ollama, then:
#     ollama pull llava          (or qwen2.5vl / moondream)
#     ollama pull llama3.1       (or any instruct model you have room for)
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "moondream"      # used for multimodal-LLM OCR (Condition 2/4)
                                 # smaller/faster than llava on CPU; swap back
                                 # to "llava" or try "qwen2.5vl" if you want to
                                 # trade speed for accuracy
TEXT_MODEL = "llama3.1"         # used for entity extraction/correction

REQUEST_TIMEOUT_SECONDS = 300

# Was "30m" (avoid reloading between calls). Changed to unload quickly
# instead: on this machine, Ollama holding both models resident (~7GB)
# continuously was starving EasyOCR's basic-OCR subprocess of memory
# (observed: ~1.8GB free out of 16.6GB total, causing EasyOCR to crash on
# ~90% of receipts). A short keep_alive means Ollama frees that memory
# between calls, trading a reload cost (seconds, not the ~70s cold-start
# worst case) for basic-OCR actually being able to run reliably.
OLLAMA_KEEP_ALIVE = "5s"

# ---------------------------------------------------------------------------
# Which condition the Gradio demo app uses by default.
# One of: "raw_ocr", "raw_ocr_llm", "multimodal_ocr", "multimodal_ocr_llm"
# ---------------------------------------------------------------------------
APP_DEFAULT_CONDITION = "raw_ocr_llm"
