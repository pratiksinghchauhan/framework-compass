"""Gradio UI for asking questions and inspecting what retrieval returned.

    python -m app.ui                  # http://127.0.0.1:7860
    python -m app.ui --host 0.0.0.0 --port 7860

The settings panel exposes the same switches as the CLI (chunk strategy,
retrieval mode, ID lookup, number of passages), so each part can be tried on
its own. The passages panel shows every retrieved chunk with its BM25 and
dense ranks and whether the answer cited it.
"""

from __future__ import annotations

import argparse
import html
from functools import cache

import gradio as gr

from app.answer import answer, source_label
from app.llm import ChatModel, get_chat_model
from app.retrieval import MODES, Hit, Retriever
from ingest.chunk import STRATEGIES

EXAMPLES = [
    "Which OWASP risk covers prompt injection?",
    "What does GOVERN 1.1 require?",
    "What is AML.T0051 and how can it be mitigated?",
    "Which GenAI Profile actions address confabulation?",
    "What is the capital of France?",
]

TABLE_HEADERS = ["#", "cited", "rrf", "bm25 rank", "dense rank", "type", "source"]


@cache
def retriever(strategy: str) -> Retriever:
    return Retriever(strategy)


_model: ChatModel | None = None


def chat_model() -> ChatModel:
    global _model
    if _model is None:
        _model = get_chat_model()
    return _model


def passages_table(hits: list[Hit], cited: list[int]) -> list[list]:
    return [[n, "yes" if n in cited else "", round(h.score, 4), h.bm25_rank or "-", h.dense_rank or "-",
             h.chunk["chunk_type"], source_label(h.chunk)]
            for n, h in enumerate(hits, start=1)]


def passages_html(hits: list[Hit], cited: list[int]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        mark = " (cited)" if n in cited else ""
        blocks.append(
            f"<details><summary><b>[{n}]{mark}</b> {html.escape(source_label(hit.chunk))}</summary>"
            f"<pre style='white-space: pre-wrap'>{html.escape(hit.chunk['text'])}</pre></details>")
    return "\n".join(blocks)


def ask(question: str, strategy: str, mode: str, id_lookup: bool, k: int, retrieve_only: bool):
    question = question.strip()
    if not question:
        return "Type a question first.", "", [], ""
    found = retriever(strategy)
    k = int(k)
    settings = f"retrieval: {mode}{', ID lookup' if mode == 'hybrid' and id_lookup else ''}, {strategy} chunks, k={k}"

    if retrieve_only:
        hits = found.search(question, k=k, mode=mode, id_lookup=id_lookup)
        return f"*Retrieve only, no model call.* {settings}", "", passages_table(hits, []), passages_html(hits, [])

    try:
        model = chat_model()
        result = answer(question, found, model, k=k, mode=mode, id_lookup=id_lookup)
    except Exception as error:  # missing key, network, quota: show it and still show the passages
        hits = found.search(question, k=k, mode=mode, id_lookup=id_lookup)
        message = f"**Model call failed:** {html.escape(str(error))}\n\nThe retrieved passages are shown below."
        return message, "", passages_table(hits, []), passages_html(hits, [])

    status = "answered" if result.covered else "not covered"
    text = f"{result.text}\n\n---\n*Model: {result.model} · {status} · {settings}*"
    sources = "\n".join(f"- **[{n}]** {source_label(result.hits[n - 1].chunk)}" for n in result.cited)
    if not sources and result.covered:
        sources = "*The answer cites no passages.*"
    return text, sources, passages_table(result.hits, result.cited), passages_html(result.hits, result.cited)


def build() -> gr.Blocks:
    with gr.Blocks(title="Framework Compass") as demo:
        gr.Markdown("# Framework Compass\nAsk the AI risk and security frameworks (NIST AI RMF, "
                    "NIST GenAI Profile, OWASP, MITRE ATLAS). Answers cite the passages they use. "
                    "Unofficial; not affiliated with NIST, OWASP or MITRE.")
        with gr.Row():
            with gr.Column(scale=3):
                question = gr.Textbox(label="Question", lines=2, autofocus=True,
                                      placeholder="e.g. What does the GenAI Profile suggest for GOVERN 1.1?")
                with gr.Row():
                    ask_button = gr.Button("Ask", variant="primary")
                    clear_button = gr.ClearButton(value="Clear")
            with gr.Column(scale=2):
                with gr.Accordion("Settings", open=True):
                    strategy = gr.Dropdown(sorted(STRATEGIES), value="structured", label="Chunks")
                    mode = gr.Radio(list(MODES), value="hybrid", label="Retrieval mode")
                    id_lookup = gr.Checkbox(value=True, label="Exact ID lookup (hybrid only)")
                    k = gr.Slider(1, 20, value=8, step=1, label="Passages given to the model (k)")
                    retrieve_only = gr.Checkbox(value=False, label="Retrieve only (no model call)")
        gr.Examples(EXAMPLES, inputs=question)

        answer_box = gr.Markdown(label="Answer")
        sources_box = gr.Markdown(label="Cited sources")
        with gr.Accordion("Retrieved passages", open=False):
            table = gr.Dataframe(headers=TABLE_HEADERS, interactive=False, wrap=True)
            texts = gr.HTML()

        inputs = [question, strategy, mode, id_lookup, k, retrieve_only]
        outputs = [answer_box, sources_box, table, texts]
        ask_button.click(ask, inputs, outputs)
        question.submit(ask, inputs, outputs)
        clear_button.add([question, answer_box, sources_box, table, texts])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gradio UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a temporary public Gradio link")
    args = parser.parse_args()
    print("Loading chunks and the embedding model...")
    retriever("structured").search("warm up", k=1)  # loads BM25, vectors and the encoder before the first question
    build().launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
