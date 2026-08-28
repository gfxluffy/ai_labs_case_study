# Receipt Data Extraction Pipeline (SROIE V2)

Implements Challenge 2: an OCR + LLM pipeline that extracts structured data
(company, date, address, total) from receipt images, evaluated across four
conditions, plus a Gradio app for interactive use.

## Project layout

```
config.py             All paths and model settings -- edit this first.
sroie_dataset.py       Dataset loading + text normalization for comparison.
ocr_basic.py            Basic OCR engine (Condition 1/3 input).
ocr_basic_worker.py     Subprocess entry point used by ocr_basic.IsolatedBasicOCR.
ocr_multimodal.py       Vision-LLM OCR via local Ollama (Condition 2/4 input).
entity_extraction.py    Entity-extraction/rectification LLM + prompt template.
pipeline.py              Orchestrates the 4 conditions.
evaluation.py            Scoring: extraction rate, exact match, fuzzy similarity.
run_evaluation.py       Script: runs all 4 conditions over the dataset, writes output/.
app.py                   Gradio demo: upload + Q&A.
requirements.txt
data/sroie_v2/           SROIE V2 dataset (train/ + test/) -- see Setup below.
output/                  Evaluation results -- created by run_evaluation.py.
```

## The four conditions

| Condition | OCR source | Entity structuring |
|---|---|---|
| `raw_ocr` | basic OCR engine | none (regex heuristic only) |
| `raw_ocr_llm` | basic OCR engine | entity-analysis LLM |
| `multimodal_ocr` | vision LLM transcription | none (regex heuristic only) |
| `multimodal_ocr_llm` | vision LLM transcription | entity-analysis LLM |

`raw_ocr` and `multimodal_ocr` intentionally use a dumb regex-based field
guesser instead of an LLM, so the "without LLM post-processing" arm of the
comparison is genuinely without LLM help. Otherwise conditions 1 and 3 (or
2 and 4) would be identical.

## Setup

1. **Python deps:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ollama (local LLM, free, no API cost):**
   Install from https://ollama.com, then:
   ```bash
   ollama pull moondream    # vision model, or swap for llava / qwen2.5vl in config.py
   ollama pull llama3.1     # text model for entity extraction + Q&A
   ollama serve             # if not already running as a service
   ```

3. **Dataset:** download SROIE V2 and unzip it into `data/sroie_v2/` (or point
   `config.py`'s `DATASET_ROOT` at wherever you put it). Expected layout:
   ```
   data/sroie_v2/
       train/img/*.jpg
       train/entities/*.txt      (JSON: company, date, address, total)
       test/img/*.jpg
       test/entities/*.txt
   ```
   If your download uses different folder names, only `config.py` needs to change.

## Running

**Smoke test (20 receipts, fast, good for checking everything's wired up):**
```bash
python run_evaluation.py --limit 20
```

**Full evaluation (test set only):**
```bash
python run_evaluation.py
```

**Larger evaluation (train+test pooled, random sample):** since nothing is
trained anywhere in this pipeline (OCR engines and LLMs are all
frozen/pretrained), pooling train + test for a bigger sample carries no
leakage risk:
```bash
python run_evaluation.py --pool all --sample 200
```

**Resuming an interrupted run** (crash, power loss, closed terminal) --
re-run the exact same command with `--resume` added; it skips receipts
already completed in `output/predictions_raw.csv`:
```bash
python run_evaluation.py --pool all --sample 200 --resume
```

Any of the above writes `output/comparison_table.csv` (the headline
4-condition comparison), `output/predictions_raw.csv` (every prediction +
ground truth),
`output/errors.csv` (receipts that failed, with reason + timing), and
`output/per_field/*.csv` (per-field breakdown per condition).
`predictions_raw.csv` and `errors.csv` are written incrementally as the run
progresses, so a long run can be checked or safely interrupted partway
through without losing completed work.

**Interactive app:**
```bash
python app.py
```

## Other Notes

- **Cost/runtime constraint:** everything here runs locally via Ollama, no
  paid API calls. Worth flagging as a discussion point: local vision models
  are slower and somewhat less accurate than hosted multimodal APIs, which
  is a real trade-off in the Condition 2/4 results, not just a footnote.
- **Rectification vs. completion:** `entity_extraction.py`'s prompt
  explicitly separates "fixing garbled OCR" from "completing truncated
  entities" per the assignment's wording -- worth pointing at directly when
  asked how you addressed that requirement.
- **AI tool disclosure:** this entire codebase was scaffolded with Claude
  (Anthropic). Document that per the assignment's requirement to disclose
  AI tool use, and note anything you changed or verified afterward.
- **Resource constraints on CPU-only hardware:** on a memory-constrained,
  no-GPU machine, expect basic OCR (EasyOCR) to fail on very large receipt
  images unless downscaled first (already handled in `ocr_basic.py`), and
  expect Ollama holding multiple models resident to compete with EasyOCR for
  RAM -- `config.py`'s `OLLAMA_KEEP_ALIVE` trades a per-call reload cost for
  headroom to avoid that. `ocr_basic.IsolatedBasicOCR` runs basic OCR in a
  subprocess so a crash there only fails one receipt instead of the whole run.
- Before running a full evaluation, run the `--limit 20` smoke test and
  eyeball `output/predictions_raw.csv` -- if the vision model is too slow
  or inaccurate on your hardware, that's worth knowing early and either
  swapping to a smaller model (moondream) or evaluating on a documented
  subsample (`--sample N`), which the assignment leaves open to you as long
  as you state it.
