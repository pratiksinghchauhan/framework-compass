# Framework Compass

**Ask the AI security frameworks. Get cited answers.**

Framework Compass is a retrieval-augmented generation (RAG) assistant for the main AI risk and security frameworks. Every answer cites the document, section and page it came from. When the frameworks don't cover a question, it says so instead of guessing.

The project ships with its own **evaluation report**: a hand-written test set, an LLM judge checked against human labels, and results with confidence intervals.

> Unofficial project. Not affiliated with or endorsed by NIST, OWASP or MITRE.

## Documents covered

| Document | Publisher | Licence |
|---|---|---|
| AI Risk Management Framework 1.0 (AI 100-1) | NIST | Public domain (US government work) |
| Generative AI Profile (AI 600-1) | NIST | Public domain (US government work) |
| Top 10 for LLM Applications 2025 | OWASP GenAI Security Project | CC BY-SA 4.0 |
| Agentic AI – Threats and Mitigations | OWASP GenAI Security Project | CC BY-SA 4.0 |
| ATLAS (optional) | MITRE | See MITRE ATLAS terms |

The source documents are not stored in this repository. See [Getting the documents](#getting-the-documents).

## How it works

1. **Ingest:** extract text from each document with section and page metadata, then split it into chunks along headings.
2. **Retrieve:** hybrid search that combines BM25 (good for exact identifiers like `LLM01` or `MAP 1.1`) with embeddings (`BAAI/bge-small-en-v1.5`, run locally).
3. **Answer:** Claude answers from the retrieved passages only, with citations to the source text.
4. **Evaluate:** an [Inspect](https://inspect.aisi.org.uk/) evaluation with single-document, multi-document and unanswerable questions.

## Project layout

```
ingest/          text extraction and chunking
app/             retrieval, answering and the Gradio UI
eval/            test set, Inspect tasks and results
data/raw/        source documents (not committed)
data/processed/  chunks and indexes (not committed)
```

## Getting the documents

Download the NIST PDFs and the ATLAS data:

```bash
python ingest/download.py
```

The OWASP PDFs must be downloaded by hand from a browser and saved in `data/raw/`:

- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/) → `data/raw/owasp-llm-top10-2025.pdf`
- [OWASP Agentic AI – Threats and Mitigations](https://genai.owasp.org/resource/agentic-ai-threats-and-mitigations/) → `data/raw/owasp-agentic-threats.pdf`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...   # never commit this
```

## Status

Early development. Ingestion, retrieval, the UI and the evaluation report are still to come.

## Credits

The OWASP Top 10 for LLM Applications and the Agentic AI Threats and Mitigations guide are © OWASP Foundation, licensed under CC BY-SA 4.0. NIST AI 100-1 and AI 600-1 are works of the US government. MITRE ATLAS is © The MITRE Corporation.
