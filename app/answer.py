"""Answer a question from retrieved passages, with numbered citations.

    python -m app.answer "Which OWASP risk covers prompt injection?"
    python -m app.answer "..." --mode bm25 --strategy generic --k 8
    python -m app.answer "..." --retrieve-only     # no model call, no API key needed
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

from app.llm import ChatModel, get_chat_model
from app.retrieval import MODES, Hit, Retriever
from ingest.chunk import DOCS, STRATEGIES

NOT_COVERED = "The frameworks in this corpus don't cover this."

SYSTEM_PROMPT = f"""You answer questions about AI risk and security frameworks \
(NIST AI RMF, NIST Generative AI Profile, OWASP, MITRE ATLAS) using only the \
numbered passages provided.

Rules:
- Use only facts stated in the passages. Do not add outside knowledge.
- Cite every claim with the passage number in square brackets, like [2] or [1][4].
- Quote framework identifiers (such as GV-1.1-001, MAP 1.1, AML.T0051, LLM01) exactly as written.
- If the passages do not answer the question, reply with exactly: {NOT_COVERED}
- If they answer only part of it, answer that part and say which part is not covered.
- Be concise."""


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    cited: list[int]  # 1-based passage numbers the answer cites
    covered: bool
    model: str


def source_label(chunk: dict) -> str:
    name = DOCS.get(chunk["doc"], {}).get("name", chunk["doc"])
    parts = [name]
    path = chunk.get("section_path") or []
    if chunk.get("chunk_type") == "action" and chunk.get("ids"):
        parts.append(chunk["ids"][0])
    elif path:
        section = path[-1]
        if len(path) > 1 and (section in ("Procedure", "Footnotes") or section.startswith("Applied to")):
            section = f"{path[-2]} › {section}"
        parts.append(section if len(section) <= 90 else section[:87].rstrip() + "...")
    start, end = chunk.get("page_start"), chunk.get("page_end")
    if start:
        parts.append(f"p. {start}" if start == end or not end else f"pp. {start}-{end}")
    url = chunk.get("meta", {}).get("url")
    if url:
        parts.append(url)
    return ", ".join(parts)


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        blocks.append(f"[{n}] Source: {source_label(hit.chunk)}\n{hit.chunk['text']}")
    return "\n\n".join(blocks)


def answer(question: str, retriever: Retriever, model: ChatModel, k: int = 8,
           mode: str = "hybrid", id_lookup: bool = True) -> Answer:
    hits = retriever.search(question, k=k, mode=mode, id_lookup=id_lookup)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Passages:\n\n{format_passages(hits)}\n\nQuestion: {question}"},
    ]
    text = model.complete(messages).strip()
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    covered = NOT_COVERED.lower().rstrip(".") not in text.lower()
    return Answer(question, text, hits, cited, covered, model.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the frameworks a question.")
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="structured")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("--no-id-lookup", action="store_true", help="hybrid without exact ID lookup")
    parser.add_argument("--k", type=int, default=8, help="passages given to the model")
    parser.add_argument("--retrieve-only", action="store_true", help="show passages, skip the model call")
    args = parser.parse_args()

    retriever = Retriever(args.strategy)
    if args.retrieve_only:
        for n, hit in enumerate(retriever.search(args.question, k=args.k, mode=args.mode,
                                                      id_lookup=not args.no_id_lookup), start=1):
            print(f"[{n}] rrf={hit.score:.4f} bm25#{hit.bm25_rank or '-'} dense#{hit.dense_rank or '-'}  "
                  f"{source_label(hit.chunk)}")
        return

    result = answer(args.question, retriever, get_chat_model(), k=args.k, mode=args.mode,
                    id_lookup=not args.no_id_lookup)
    print(result.text)
    print(f"\nModel: {result.model} | retrieval: {args.mode}, {args.strategy}, k={args.k}")
    for n in result.cited:
        print(f"  [{n}] {source_label(result.hits[n - 1].chunk)}")


if __name__ == "__main__":
    main()
