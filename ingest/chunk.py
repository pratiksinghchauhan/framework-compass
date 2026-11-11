"""Split the source documents into retrieval chunks.

Chunks follow each document's own structure: one chunk per NIST subcategory
or suggested action, per OWASP entry subsection, per ATLAS object. Only prose
sections are packed into token-limited chunks, with a small sentence overlap.
Every chunk starts with a header naming its document and section path.

The "generic" strategy is a baseline for comparison: fixed token windows over
the raw extracted text, with no cleanup, structure or headers.

    python -m ingest.chunk [--strategy structured|generic] [--target 400]
                           [--max-tokens 512] [--overlap 50]

Writes data/processed/<strategy>/chunks.jsonl and chunk_stats.json.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pypdf import PdfReader

from ingest import atlas
from ingest.pdf_text import Line, Section, read_sections

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# The model expects these task prefixes on the text it embeds.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

ID_PATTERNS = [
    r"\bAML\.(?:TA|T|M|CS|E|S)\d{4}(?:\.\d{3})?\b",
    r"\b(?:GV|MP|MS|MG)-\d+\.\d+-\d{3}\b",
    r"\b(?:GOVERN|MAP|MEASURE|MANAGE) \d+(?:\.\d+)?\b",
    r"\bLLM\d{2}\b",
]

# Kinds that are one complete unit of the source. They are kept whole unless
# they exceed the token limit; prose is packed to the target size instead.
ATOMIC = {
    "category", "subcategory", "action", "footnote", "tactic", "technique", "mitigation",
    "mitigation-use", "case-study", "entity", "software",
}

GAI_RISKS = [
    "CBRN Information or Capabilities", "Confabulation",
    "Dangerous, Violent, or Hateful Content", "Data Privacy",
    "Environmental Impacts", "Harmful Bias and Homogenization",
    "Human-AI Configuration", "Information Integrity", "Information Security",
    "Intellectual Property", "Obscene, Degrading, and/or Abusive Content",
    "Value Chain and Component Integration",
]


@dataclass
class Para:
    text: str
    page_start: str | None = None
    page_end: str | None = None


@dataclass
class Unit:
    kind: str
    path: list[str]
    paras: list[Para]
    primary_id: str | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Chunk:
    id: str
    doc: str
    doc_title: str
    section_path: list[str]
    chunk_type: str
    page_start: str | None
    page_end: str | None
    ids: list[str]
    header: str
    body: str
    text: str
    n_tokens: int
    part: int
    parts: int
    meta: dict


class Tokens:
    def __init__(self, name: str):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.model_max_length = 10**9

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def windows(self, text: str, size: int) -> list[str]:
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        return [
            text[offsets[i][0]:offsets[min(i + size, len(offsets)) - 1][1]].strip()
            for i in range(0, len(offsets), size)
        ]


# ---------------------------------------------------------------- text helpers

SENTENCE_END = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[\"“(•A-Z0-9])")
BULLET = re.compile(r"^[•◦▪●\-–]\s")


def join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-"):
            out += line
        else:
            out += " " + line
    return out.strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def paragraphs(lines: list[Line]) -> list[Para]:
    """Group wrapped lines into paragraphs.

    A new paragraph starts at a bullet, or after a line that ends a sentence
    and is noticeably shorter than a full line of text.
    """
    if not lines:
        return []
    full = statistics.quantiles([len(l.text) for l in lines], n=10)[-1] if len(lines) > 2 else 80
    paras, current = [], []

    def close():
        if current:
            paras.append(Para(join_lines([l.text for l in current]),
                              current[0].page, current[-1].page))

    for line in lines:
        if current:
            prev = current[-1].text
            ends = re.search(r"[.!?:;]['\"”’)]*$", prev)
            if BULLET.match(line.text) or (ends and len(prev) < 0.8 * full):
                close()
                current = []
        current.append(line)
    close()
    return paras


# ------------------------------------------------------- document-specific units

CORE_LINE = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) (\d+)(\.\d+)?:\s*")
TABLE_CAPTION = re.compile(r"^Table \d+:")
TABLE_COLUMNS = "Categories Subcategories"


def units_rmf_core(section: Section) -> list[Unit]:
    """NIST AI 100-1 Core tables: one unit per category and per subcategory."""
    if not any(CORE_LINE.match(l.text) for l in section.lines):
        return [Unit("prose", section.path, paragraphs(section.lines))]
    prose, entries = [], []  # entries: [kind, ident, lines, category_index]
    category = None
    for line in section.lines:
        if TABLE_CAPTION.match(line.text) or line.text == TABLE_COLUMNS:
            continue
        m = CORE_LINE.match(line.text)
        if m:
            ident = f"{m[1]} {m[2]}{m[3] or ''}"
            if m[3] is None:
                category = len(entries)
                entries.append(["category", ident, [line], None])
            else:
                entries.append(["subcategory", ident, [line], category])
        elif entries:
            entries[-1][2].append(line)
        else:
            prose.append(line)

    units = [Unit("prose", section.path, paragraphs(prose))] if prose else []
    texts = [join_lines([l.text for l in lines]) for _, _, lines, _ in entries]
    for k, (kind, ident, lines, parent) in enumerate(entries):
        path = list(section.path)
        meta = {}
        if kind == "subcategory" and parent is not None:
            path.append(texts[parent])
            meta["category"] = entries[parent][1]
        if kind == "category":
            subs = [e[1] for e in entries if e[3] == k]
            text = texts[k] + (f" Subcategories: {', '.join(subs)}." if subs else "")
        else:
            text = texts[k]
        units.append(Unit(kind, path, [Para(text, lines[0].page, lines[-1].page)], ident, meta))
    return units


SUBCAT_LINE = re.compile(r"^(GOVERN|MAP|MEASURE|MANAGE) (\d+\.\d+):")
ACTION_LINE = re.compile(r"^((?:GV|MP|MS|MG)-\d+\.\d+-\d{3})\b\s*(.*)$")
ACTORS_LINE = re.compile(r"^AI Actors?(?: Tasks?)?:\s*(.*)$")
COLUMN_HEADER = "Action ID Suggested Action GAI Risks"
# Short or variant tags used in a few table rows of AI 600-1.
RISK_ALIASES = {"Environmental": "Environmental Impacts", "Human AI Configuration": "Human-AI Configuration"}
_RISK = "|".join(
    re.escape(r).replace(r"\ ", r"\s+").replace(r"\-", r"-\s*")
    for r in sorted([*GAI_RISKS, *RISK_ALIASES], key=len, reverse=True)
)
RISK_START = re.compile(rf"(?:{_RISK})")


def split_risks(text: str) -> tuple[str, list[str]]:
    """Separate the trailing GAI Risks column from an action's text.

    The column is a ';'-separated list that starts with one of the 12 named
    risks. A few rows add other short tags ("Civil Rights violations") or have
    a short form ("Environmental"); aliases are mapped to the named risk and
    other tags are kept as given.
    """
    def item_ok(item: str) -> bool:
        return bool(item) and "." not in item and len(item.split()) <= 5

    for m in RISK_START.finditer(text):
        items = [re.sub(r"\s+", " ", i).replace("- ", "-").strip() for i in text[m.start():].split(";")]
        if all(item_ok(i) for i in items):
            return text[:m.start()].strip(), [RISK_ALIASES.get(i, i) for i in items]
    return text, []


def units_gai_actions(section: Section) -> list[Unit]:
    """NIST AI 600-1 suggested-action tables: one unit per subcategory and per action."""
    if not any(ACTION_LINE.match(l.text) for l in section.lines):
        return [Unit("prose", section.path, paragraphs(section.lines))]
    prose, subcats = [], []  # subcat: dict(ident, lines, actors, actions=[(id, lines)])
    mode = "prose"
    for line in section.lines:
        text = line.text
        if text == COLUMN_HEADER:
            continue
        if m := SUBCAT_LINE.match(text):
            subcats.append({"ident": f"{m[1]} {m[2]}", "lines": [line], "actors": [], "actions": []})
            mode = "subcat"
        elif (m := ACTION_LINE.match(text)) and subcats:
            first = [Line(m[2], line.page_index, line.page)] if m[2] else []
            subcats[-1]["actions"].append((m[1], first, line.page))
            mode = "action"
        elif (m := ACTORS_LINE.match(text)) and subcats:
            subcats[-1]["actors"].append(m[1])
            mode = "actors"
        elif mode == "prose":
            prose.append(line)
        elif mode == "subcat":
            subcats[-1]["lines"].append(line)
        elif mode == "action":
            subcats[-1]["actions"][-1][1].append(line)
        else:
            subcats[-1]["actors"].append(text)

    units = [Unit("prose", section.path, paragraphs(prose))] if prose else []
    for sub in subcats:
        heading = join_lines([l.text for l in sub["lines"]])
        actors = join_lines(sub["actors"])
        action_ids = [a[0] for a in sub["actions"]]
        summary = heading
        if actors:
            summary += f"\nAI Actor Tasks: {actors}"
        if action_ids:
            summary += f"\nSuggested actions: {', '.join(action_ids)}."
        page0, page1 = sub["lines"][0].page, sub["lines"][-1].page
        units.append(Unit("subcategory", section.path, [Para(summary, page0, page1)],
                          sub["ident"], {"ai_actor_tasks": actors}))
        for action_id, lines, page in sub["actions"]:
            text = join_lines([l.text for l in lines])
            text, risks = split_risks(text)
            body = f"{action_id}: {text}"
            if risks:
                body += f"\nGAI risks: {'; '.join(risks)}."
            pages = [page] + [l.page for l in lines]
            units.append(Unit("action", section.path + [heading], [Para(body, pages[0], pages[-1])],
                              action_id, {"subcategory": sub["ident"], "gai_risks": risks,
                                          "ai_actor_tasks": actors}))
    return units


def units_default(section: Section) -> list[Unit]:
    return [Unit("prose", section.path, paragraphs(section.lines))]


DOCS = {
    "nist-ai-100-1": {
        "file": "nist-ai-100-1.pdf",
        "name": "NIST AI 100-1",
        "title": "NIST AI 100-1: Artificial Intelligence Risk Management Framework (AI RMF 1.0)",
        "units": units_rmf_core,
    },
    "nist-ai-600-1": {
        "file": "nist-ai-600-1.pdf",
        "name": "NIST AI 600-1",
        "title": "NIST AI 600-1: Generative Artificial Intelligence Profile",
        "units": units_gai_actions,
        "skip": [r"References$"],
        "footnotes": True,
    },
    "owasp-llm-top10-2025": {
        "file": "owasp-llm-top10-2025.pdf",
        "name": "OWASP Top 10 for LLM Applications 2025",
        "title": "OWASP Top 10 for LLM Applications 2025",
        "skip": [r"^Reference Links$"],
        "headings": [
            (0, r"LLM\d{2}(?::\d{4})?\s+\S.*"),
            (1, r"Description|Common Examples of (?:Vulnerability|Risk)|Prevention and Mitigation "
                r"Strategies|Example Attack Scenarios|Related Frameworks and Taxonomies|Reference Links"),
        ],
    },
    "owasp-agentic-threats": {
        "file": "owasp-agentic-threats.pdf",
        "name": "OWASP Agentic AI Threats and Mitigations",
        "title": "OWASP Agentic AI: Threats and Mitigations",
        "ids": [r"\bT\d{1,2}\b"],
    },
    "mitre-atlas": {
        "file": "mitre-atlas.yaml",
        "name": "MITRE ATLAS",
        "title": "MITRE ATLAS",
    },
}


# ------------------------------------------------------------------- packing

@dataclass
class Seg:
    text: str
    n: int
    page_start: str | None
    page_end: str | None
    para_start: bool


class Packer:
    def __init__(self, tokens: Tokens, target: int, max_tokens: int, overlap: int):
        self.tokens, self.target, self.max_tokens, self.overlap = tokens, target, max_tokens, overlap

    def budget(self, header: str) -> int:
        # [CLS] + document prefix + header + "\n" + body + [SEP], with a small
        # margin because token counts of joined text can differ slightly from
        # the sum of parts.
        return self.max_tokens - self.tokens.count(DOCUMENT_PREFIX + header + "\n") - 2 - 8

    def segments(self, paras: list[Para], limit: int, budget: int) -> list[Seg]:
        segs = []
        for para in paras:
            n = self.tokens.count(para.text)
            if n <= limit:
                segs.append(Seg(para.text, n, para.page_start, para.page_end, True))
                continue
            first = True
            for sentence in split_sentences(para.text):
                pieces = [sentence]
                if self.tokens.count(sentence) > budget:
                    pieces = self.tokens.windows(sentence, budget)
                for piece in pieces:
                    segs.append(Seg(piece, self.tokens.count(piece),
                                    para.page_start, para.page_end, first))
                    first = False
        return segs

    def tail(self, segs: list[Seg]) -> list[Seg]:
        """Whole trailing sentences of a chunk, at most `overlap` tokens."""
        out, total = [], 0
        for seg in reversed(segs):
            for sentence in reversed(split_sentences(seg.text)):
                n = self.tokens.count(sentence)
                if total + n > self.overlap:
                    return out
                out.insert(0, Seg(sentence, n, seg.page_start, seg.page_end, False))
                total += n
        return out

    def pack(self, paras: list[Para], header: str, atomic: bool) -> list[list[Seg]]:
        budget = self.budget(header)
        target = budget if atomic else min(self.target, budget)
        segs = self.segments(paras, target, budget)
        if atomic and sum(s.n for s in segs) <= budget:
            return [segs]
        chunks, current, overlap_count = [], [], 0
        for seg in segs:
            if current and sum(s.n for s in current) + seg.n > target:
                chunks.append((current, overlap_count))
                carry = self.tail(current) if self.overlap else []
                if sum(s.n for s in carry) + seg.n > budget:
                    carry = []
                current, overlap_count = list(carry), len(carry)
            current.append(seg)
        if current:
            chunks.append((current, overlap_count))
        # Fold a small last piece into the previous chunk when it fits.
        if len(chunks) > 1:
            last, carried = chunks[-1]
            new = last[carried:]
            prev = chunks[-2][0]
            if sum(s.n for s in new) < 60 and sum(s.n for s in prev + new) <= budget:
                chunks[-2] = (prev + new, chunks[-2][1])
                chunks.pop()
        return [c for c, _ in chunks]


def render(segs: list[Seg]) -> str:
    out = ""
    for seg in segs:
        sep = "\n" if seg.para_start else " "
        out = seg.text if not out else out + sep + seg.text
    return out


# ----------------------------------------------------------------------- main

def find_ids(text: str, patterns: list[str]) -> list[str]:
    seen = []
    for pattern in patterns:
        for match in re.findall(pattern, text):
            if match not in seen:
                seen.append(match)
    return seen


FOOTNOTE_START = re.compile(r"^\d{1,3} \S")


def load_units(doc: str, config: dict, warnings: list[str]) -> list[Unit]:
    path = RAW_DIR / config["file"]
    if doc == "mitre-atlas":
        return [Unit(u["kind"], u["path"], [Para(p) for p in u["paras"]], u["primary_id"], u["meta"])
                for u in atlas.load_units(path)]
    sections = read_sections(path, skip=config.get("skip", ()), patterns=config.get("headings", ()),
                             footnotes=config.get("footnotes", False), warnings=warnings)
    make = config.get("units", units_default)
    units = []
    for section in sections:
        notes = [l for l in section.lines if l.footnote]
        section.lines = [l for l in section.lines if not l.footnote]
        units += make(section)
        for note in footnote_groups(notes):
            units.append(Unit("footnote", section.path + ["Footnotes"],
                              [Para(join_lines([l.text for l in note]), note[0].page, note[-1].page)]))
    return [u for u in units if u.paras]


def footnote_groups(lines: list[Line]) -> list[list[Line]]:
    """One group per footnote: a numbered line plus its continuation lines."""
    groups = []
    for line in lines:
        if not groups or FOOTNOTE_START.match(line.text):
            groups.append([])
        groups[-1].append(line)
    return groups


def chunk_doc(doc: str, config: dict, packer: Packer, warnings: list[str]) -> list[Chunk]:
    patterns = ID_PATTERNS + config.get("ids", [])
    chunks = []
    for unit in load_units(doc, config, warnings):
        header = "[" + " › ".join([config["name"], *unit.path]) + "]"
        pieces = packer.pack(unit.paras, header, atomic=unit.kind in ATOMIC)
        for part, segs in enumerate(pieces, start=1):
            body = render(segs)
            text = f"{header}\n{body}"
            ids = find_ids(text, patterns)
            if unit.primary_id:
                ids = [unit.primary_id] + [i for i in ids if i != unit.primary_id]
            pages = [p for s in segs for p in (s.page_start, s.page_end) if p]
            chunks.append(Chunk(
                id=f"{doc}:{len(chunks):04d}",
                doc=doc,
                doc_title=config["title"],
                section_path=unit.path,
                chunk_type=unit.kind,
                page_start=pages[0] if pages else None,
                page_end=pages[-1] if pages else None,
                ids=ids,
                header=header,
                body=body,
                text=text,
                n_tokens=packer.tokens.count(text) + 2,
                part=part,
                parts=len(pieces),
                meta=unit.meta,
            ))
    return chunks


def generic_doc(doc: str, config: dict, packer: Packer, warnings: list[str]) -> list[Chunk]:
    """Baseline: fixed token windows over the raw text, as most RAG tutorials do.

    PDFs use pypdf's text as extracted, with no cleanup, no sections and no
    headers. The ATLAS YAML is windowed as plain text. Windows are `target`
    tokens long and consecutive windows share `overlap` tokens.
    """
    path = RAW_DIR / config["file"]
    if path.suffix == ".pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    else:
        pages = [path.read_text(encoding="utf-8")]
    text, starts = "", []
    for page in pages:
        starts.append(len(text))
        text += page + "\n"

    size = min(packer.target, packer.max_tokens - 2)
    step = max(1, size - packer.overlap)
    enc = packer.tokens.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    patterns = ID_PATTERNS + config.get("ids", [])
    chunks = []

    def page_of(char: int) -> str | None:
        return str(bisect.bisect_right(starts, char)) if path.suffix == ".pdf" else None

    for i in range(0, len(offsets), step):
        j = min(i + size, len(offsets))
        start, end = offsets[i][0], offsets[j - 1][1]
        body = text[start:end].strip()
        if body:
            chunks.append(Chunk(
                id=f"{doc}:{len(chunks):04d}", doc=doc, doc_title=config["title"],
                section_path=[], chunk_type="window",
                page_start=page_of(start), page_end=page_of(end - 1),
                ids=find_ids(body, patterns), header="", body=body, text=body,
                n_tokens=packer.tokens.count(body) + 2, part=1, parts=1, meta={},
            ))
        if j == len(offsets):
            break
    return chunks


STRATEGIES = {"structured": chunk_doc, "generic": generic_doc}


def percentile(values: list[int], q: float) -> int:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def stats(chunks: list[Chunk], args, warnings: dict, missing: list[str]) -> dict:
    by_doc = defaultdict(list)
    for c in chunks:
        by_doc[c.doc].append(c)

    def summary(items):
        n = [c.n_tokens for c in items]
        return {
            "chunks": len(items),
            "tokens_p50": percentile(n, 0.5),
            "tokens_p90": percentile(n, 0.9),
            "tokens_max": max(n),
            "over_target": sum(1 for c in items if c.n_tokens > args.target),
            "with_ids": round(sum(1 for c in items if c.ids) / len(items), 3),
            "split_units": sum(1 for c in items if c.parts > 1 and c.part == 1),
        }

    return {
        "config": {"strategy": args.strategy, "tokenizer": EMBEDDING_MODEL, "target": args.target,
                   "max_tokens": args.max_tokens, "overlap": args.overlap},
        "total": summary(chunks) if chunks else {},
        "docs": {
            doc: {**summary(items), "by_type": dict(Counter(c.chunk_type for c in items))}
            for doc, items in by_doc.items()
        },
        "missing_docs": missing,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), default="structured",
                        help="structured (default) or the generic fixed-window baseline")
    parser.add_argument("--target", type=int, default=400,
                        help="target tokens for prose chunks; window size for generic")
    parser.add_argument("--max-tokens", type=int, default=512, help="hard limit per chunk, counted with the embedding model's tokenizer")
    parser.add_argument("--overlap", type=int, default=50,
                        help="max overlap tokens between prose chunks; exact overlap for generic")
    parser.add_argument("--out", type=Path, help="output folder (default data/processed/<strategy>)")
    args = parser.parse_args()
    args.out = args.out or OUT_DIR / args.strategy
    make_chunks = STRATEGIES[args.strategy]

    packer = Packer(Tokens(EMBEDDING_MODEL), args.target, args.max_tokens, args.overlap)
    chunks, warnings, missing = [], {}, []
    for doc, config in DOCS.items():
        if not (RAW_DIR / config["file"]).exists():
            missing.append(config["file"])
            print(f"missing, skipped: data/raw/{config['file']}")
            continue
        doc_warnings = []
        doc_chunks = make_chunks(doc, config, packer, doc_warnings)
        chunks += doc_chunks
        if doc_warnings:
            warnings[doc] = doc_warnings
        print(f"{doc}: {len(doc_chunks)} chunks")
        for w in doc_warnings:
            print(f"  warning: {w}")

    too_big = [c for c in chunks if c.n_tokens > args.max_tokens]
    if too_big:
        raise SystemExit(f"{len(too_big)} chunks exceed {args.max_tokens} tokens, e.g. {too_big[0].id}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    report = stats(chunks, args, warnings, missing)
    (args.out / "chunk_stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    t = report["total"]
    print(f"\n{t['chunks']} chunks, tokens p50 {t['tokens_p50']}, p90 {t['tokens_p90']}, "
          f"max {t['tokens_max']} -> {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out}")


if __name__ == "__main__":
    main()
