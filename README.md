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
2. **Retrieve:** hybrid search that combines BM25 (good for exact identifiers like `LLM01` or `MAP 1.1`), embeddings (`nomic-ai/nomic-embed-text-v1.5`, run locally) and exact identifier lookup.
3. **Answer:** Gemini answers from the retrieved passages only, with citations to the source text.
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

## Chunking

```bash
python -m ingest.chunk
```

Chunks follow each document's structure: one per NIST AI RMF subcategory, NIST GenAI Profile suggested action, OWASP Top 10 entry subsection, OWASP Agentic threat (T1–T17) and ATLAS object. Prose sections are packed to about 400 tokens with up to 50 tokens of sentence overlap, and no chunk exceeds 512 tokens, counted with the embedding model's tokenizer. Every chunk starts with a header naming its document and section, and records its pages and any framework identifiers it mentions.

For comparison, `python -m ingest.chunk --strategy generic` builds the usual tutorial baseline: fixed 400-token windows with 50 tokens of overlap over the raw extracted text, with no cleanup, sections or headers.

Each strategy writes `data/processed/<strategy>/chunks.jsonl`, with size statistics in `chunk_stats.json` next to it.

## Embeddings

```bash
python -m ingest.embed
```

Embeds every chunk with `nomic-ai/nomic-embed-text-v1.5`, using the `search_document: ` prefix the model expects. Each strategy gets `embeddings.npy` (one normalised 768-dimensional float32 row per chunk, in the same order as `chunks.jsonl`) and `embeddings.json` (model, chunk ids and a hash of the chunk file). A strategy is skipped if its embeddings already match its chunks. On a 2-core CPU, embedding both strategies takes about 40 minutes.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env       # then set GEMINI_API_KEY (from Google AI Studio)
```

`.env` is gitignored. Exporting the variables in your shell works too, and takes priority over `.env`.

## Asking questions

```bash
python -m app.answer "Which GenAI Profile actions address confabulation?"
python -m app.answer "What is AML.T0051?" --retrieve-only   # passages only, no API key needed
```

Retrieval fuses three rankings with reciprocal rank fusion: BM25, embeddings, and exact identifier lookup (if the question names `AML.T0051` or `GOVERN 1.1`, the chunk for that ID ranks first). Use `--mode bm25`, `--mode dense` or `--no-id-lookup` to test the parts on their own, and `--strategy generic` to search the baseline chunks.

Answers come from Gemini, through its OpenAI-compatible API. The key is read from `GEMINI_API_KEY`, and `LLM_MODEL` picks the Gemini model (default `gemini-3.5-flash-lite`). The model call sits behind one small interface in `app/llm.py`.

## Web UI

```bash
python -m app.ui            # then open http://127.0.0.1:7860
```

A Gradio page with the same switches as the CLI (chunks, retrieval mode, ID lookup, k, retrieve only). Under the answer it lists the cited sources, and a "Retrieved passages" panel shows every passage the model saw, with its BM25 and dense ranks, whether it was cited, and its full text. If the model call fails (for example, no API key), the passages are still shown. Use `--host 0.0.0.0` to serve on the network, or `--share` for a temporary public Gradio link.

## Status

Early development. Ingestion, retrieval, answering and the UI work. The evaluation report is still to come.

## Credits

The OWASP Top 10 for LLM Applications and the Agentic AI Threats and Mitigations guide are © OWASP Foundation, licensed under CC BY-SA 4.0. NIST AI 100-1 and AI 600-1 are works of the US government. MITRE ATLAS is © The MITRE Corporation.
