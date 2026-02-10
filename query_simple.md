# skills/pdf/scripts/query_pdf_simple.py
#
# Compatible with the "skills/pdf/scripts/" pattern above:
# - Runs with cwd = workdir/ (so inputs are relative to workdir/)
# - Uses python-only libs (pdfplumber + openai)
# - Prints JSON to stdout on success
#
# Usage examples (from inside workdir/):
# python skills/pdf/scripts/query_pdf_simple.py --in report.pdf --q "What is the total revenue in 2024?"
#
# Notes:
# - This script sends (most of) the PDF text directly to the LLM.
# - For large PDFs, it will automatically summarize in chunks first, then answer from the summary.

import argparse
import json
import os
from pathlib import Path
from typing import List

import pdfplumber
from openai import OpenAI

client = OpenAI()

DEFAULT_MODEL = "gpt-5"
# Character limits are a practical proxy for token limits.
# If your PDFs are large, the script will fall back to chunk summarization.
MAX_CONTEXT_CHARS = 120_000
CHUNK_CHARS = 18_000


def extract_all_text(pdf_path: Path) -> str:
    parts: List[str] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n\n".join(parts).strip()


def chunk_text(text: str, chunk_chars: int) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_chars)
        chunk = text[i:j]

        # Try to end on paragraph boundary for cleaner summaries
        k = chunk.rfind("\n\n")
        if k > int(chunk_chars * 0.6):
            chunk = chunk[:k]
            j = i + k

        chunk = chunk.strip()
        if chunk:
            chunks.append(chunk)

        i = j
    return chunks


def llm_summarize_chunk(model: str, chunk: str, chunk_id: int) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You summarize PDF text for downstream Q&A.\n"
                    "Return a concise, factual summary capturing key entities, numbers, dates, and definitions.\n"
                    "If tables are present in the text, summarize their key values.\n"
                    "Do not add information that is not present."
                ),
            },
            {
                "role": "user",
                "content": f"Chunk {chunk_id} text:\n{chunk}",
            },
        ],
    )
    return resp.output_text.strip()


def llm_answer_from_context(model: str, question: str, context: str) -> str:
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "Answer the user's question using ONLY the provided PDF content.\n"
                    "If the answer is not present, say you cannot find it in the provided content.\n"
                    "Be concise and include key numbers/dates when relevant."
                ),
            },
            {
                "role": "user",
                "content": f"PDF content:\n{context}\n\nQuestion:\n{question}",
            },
        ],
    )
    return resp.output_text.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract all text from a PDF and ask an LLM a question (python-only).")
    ap.add_argument("--in", dest="inp", required=True, help="Input PDF (relative to workdir/)")
    ap.add_argument("--q", dest="question", required=True, help="Question to ask")
    ap.add_argument("--model", dest="model", default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    ap.add_argument(
        "--max_context_chars",
        dest="max_context_chars",
        type=int,
        default=MAX_CONTEXT_CHARS,
        help=f"Max chars to send as raw context before summarizing (default: {MAX_CONTEXT_CHARS})",
    )
    ap.add_argument(
        "--chunk_chars",
        dest="chunk_chars",
        type=int,
        default=CHUNK_CHARS,
        help=f"Chunk size for summarization fallback (default: {CHUNK_CHARS})",
    )
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    pdf_path = Path(args.inp)
    if not pdf_path.exists():
        raise SystemExit(f"File not found: {args.inp}")

    full_text = extract_all_text(pdf_path)
    if not full_text:
        raise SystemExit("No extractable text found in this PDF (it may be scanned images).")

    used_mode = "raw"
    context = full_text

    # If too large, summarize in chunks first, then answer from combined summary
    if len(full_text) > args.max_context_chars:
        used_mode = "chunk_summarize"
        chunks = chunk_text(full_text, args.chunk_chars)
        summaries: List[str] = []
        for i, ch in enumerate(chunks, start=1):
            summaries.append(llm_summarize_chunk(args.model, ch, i))
        context = "\n\n---\n\n".join([f"[Summary Chunk {i}]\n{s}" for i, s in enumerate(summaries, start=1)])

    answer = llm_answer_from_context(args.model, args.question, context)

    print(
        json.dumps(
            {
                "ok": True,
                "pdf": args.inp,
                "mode": used_mode,
                "text_chars": len(full_text),
                "answer": answer,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
