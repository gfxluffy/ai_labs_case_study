"""
Gradio demo app.

Flow:
  1. User uploads a receipt image.
  2. The pipeline (config.APP_DEFAULT_CONDITION, i.e. best-performing
     condition from your evaluation) runs and extracts the 4 fields.
  3. User can ask free-form questions ("what's the total?", "who's the
     vendor?", "was this a restaurant receipt?") answered by a lightweight
     LLM call that's given the extracted entities (+ raw OCR text as
     backup context) rather than re-processing the image each time.

Run with:  python app.py
"""

import tempfile
from pathlib import Path

import gradio as gr
import requests

import config
from pipeline import ReceiptPipeline

pipeline = ReceiptPipeline()

QA_PROMPT_TEMPLATE = """You are answering questions about a receipt based on the extracted data below.
If the answer isn't contained in the data, say so rather than guessing.

Extracted fields:
  company: {company}
  date: {date}
  address: {address}
  total: {total}

Raw OCR text (use only if the extracted fields above don't answer the question):
---
{raw_text}
---

Question: {question}

Answer concisely in one or two sentences."""


def process_receipt(image_file):
    if image_file is None:
        empty = {"company": "", "date": "", "address": "", "total": ""}
        return empty, "Upload a receipt to begin.", empty, ""

    result = pipeline.run_single_condition(Path(image_file), config.APP_DEFAULT_CONDITION)
    entities = result["entities"]
    raw_text = result["raw_text"]

    # outputs: JSON display, raw text display, entities_state, raw_text_state
    return entities, raw_text, entities, raw_text


def answer_question(question, entities_state, raw_text_state):
    if not entities_state:
        return "Upload and process a receipt first."
    if not question or not question.strip():
        return "Ask a question about the uploaded receipt."

    prompt = QA_PROMPT_TEMPLATE.format(
        company=entities_state.get("company", "unknown"),
        date=entities_state.get("date", "unknown"),
        address=entities_state.get("address", "unknown"),
        total=entities_state.get("total", "unknown"),
        raw_text=raw_text_state or "",
        question=question,
    )

    response = requests.post(
        f"{config.OLLAMA_BASE_URL}/api/generate",
        json={"model": config.TEXT_MODEL, "prompt": prompt, "stream": False},
        timeout=config.REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


with gr.Blocks(title="Receipt Data Extraction") as demo:
    gr.Markdown("# Receipt Data Extraction\nUpload a receipt image, then ask questions about it.")

    raw_text_state = gr.State("")
    entities_state = gr.State({})

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Receipt image")
            process_btn = gr.Button("Extract data", variant="primary")

        with gr.Column():
            entities_output = gr.JSON(label="Extracted fields")
            raw_text_output = gr.Textbox(label="Raw OCR text", lines=8)

    gr.Markdown("## Ask about this receipt")
    with gr.Row():
        question_input = gr.Textbox(label="Question", placeholder="What's the total?")
        answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_btn = gr.Button("Ask")

    process_btn.click(
        fn=process_receipt,
        inputs=[image_input],
        outputs=[entities_output, raw_text_output, entities_state, raw_text_state],
    )

    ask_btn.click(
        fn=answer_question,
        inputs=[question_input, entities_state, raw_text_state],
        outputs=[answer_output],
    )

if __name__ == "__main__":
    demo.launch()
