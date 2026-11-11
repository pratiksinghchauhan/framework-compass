"""Turn a PDF into clean, page-tagged lines grouped into outline sections."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Line:
    text: str
    page_index: int
    page: str  # printed page label, falling back to the 1-based page number
    footnote: bool = False


@dataclass
class Section:
    path: list[str]
    lines: list[Line]


WORD = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")
PAGE_NUMBER = re.compile(r"^(\d+|[ivxlc]+)$", re.IGNORECASE)
NUMBERING = r"(?:(?:[A-Z]|\d+)(?:\.\d+)*\.?\s+)?"


def normalize(text: str) -> str:
    """NFKC folds ligatures such as 'ﬁ' into 'fi', so keyword search matches them."""
    text = unicodedata.normalize("NFKC", text).replace("­", "")
    return re.sub(r"[ \t]+", " ", text)


def _page_labels(reader: PdfReader) -> list[str]:
    try:
        labels = list(reader.page_labels)
    except Exception:
        labels = []
    if len(labels) != len(reader.pages):
        labels = [str(i + 1) for i in range(len(reader.pages))]
    return labels


def _strip_running_text(pages: list[list[str]]) -> list[list[str]]:
    """Drop page numbers and running headers/footers.

    A line counts as running text when it sits in the top or bottom two lines
    of a page and the same line appears there on at least a quarter of pages.
    """
    edges = Counter()
    for lines in pages:
        edges.update(set(lines[:2] + lines[-2:]))
    limit = max(3, len(pages) // 4)
    repeated = {line for line, count in edges.items() if count >= limit}

    cleaned = []
    for lines in pages:
        keep = []
        for i, line in enumerate(lines):
            at_edge = i < 2 or i >= len(lines) - 2
            if at_edge and (line in repeated or PAGE_NUMBER.match(line)):
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned


def _dehyphenate(lines: list[str], vocab: set[str]) -> list[str]:
    """Rejoin words split across lines ("or-" + "ganizational").

    Keeps the hyphen when the hyphenated form appears elsewhere in the
    document ("context-specific"), otherwise joins the halves.
    """
    out = list(lines)
    i = 0
    while i < len(out) - 1:
        cur, nxt = out[i], out[i + 1]
        if re.search(r"[a-z]-$", cur) and re.match(r"[a-z]", nxt):
            left = WORD.findall(cur)[-1]
            right, rest = re.match(r"(\S+)\s*(.*)", nxt).groups()
            core = re.match(r"[a-z]+", right).group(0)
            hyphenated, joined = f"{left}-{core}".lower(), f"{left}{core}".lower()
            if hyphenated in vocab:
                joiner = "-"
            elif joined in vocab:
                joiner = ""
            else:
                both_words = left.lower() in vocab and core in vocab and min(len(left), len(core)) > 2
                joiner = "-" if both_words else ""
            out[i] = cur[:-1] + joiner + right
            out[i + 1] = rest
            if not rest:
                del out[i + 1]
                continue
        i += 1
    return out


FOOTNOTE = re.compile(r"^(\d{1,3}) \S")


def _mark_footnotes(lines: list[Line]) -> None:
    """Flag numbered footnotes at the bottom of pages.

    A footnote starts with the next expected footnote number at the start of a
    line; it and everything after it on that page are footnote text.
    """
    expected, in_notes, page = 1, False, None
    for line in lines:
        if line.page_index != page:
            page, in_notes = line.page_index, False
        m = FOOTNOTE.match(line.text)
        if m and int(m[1]) == expected:
            in_notes, expected = True, expected + 1
        line.footnote = in_notes


def read_lines(path: Path, footnotes: bool = False) -> tuple[PdfReader, list[Line]]:
    reader = PdfReader(path)
    labels = _page_labels(reader)
    pages = []
    for page in reader.pages:
        text = normalize(page.extract_text() or "")
        pages.append([line.strip() for line in text.splitlines() if line.strip()])
    pages = _strip_running_text(pages)
    vocab = {w.lower() for lines in pages for line in lines for w in WORD.findall(line)}
    lines = []
    for index, page_lines in enumerate(pages):
        for text in _dehyphenate(page_lines, vocab):
            lines.append(Line(text, index, labels[index]))
    if footnotes:
        _mark_footnotes(lines)
    return reader, lines


def _outline(reader: PdfReader) -> list[tuple[int, str, int]]:
    entries = []

    def walk(items, depth):
        for item in items:
            if isinstance(item, list):
                walk(item, depth + 1)
                continue
            title = normalize(item.title).strip()
            if re.search(r"[A-Za-z]", title):
                entries.append((depth, title, reader.get_destination_page_number(item)))

    walk(reader.outline, 0)
    return entries


def _title_pattern(title: str) -> re.Pattern:
    words = r"\s+".join(re.escape(w) for w in title.split())
    return re.compile(rf"^{NUMBERING}{words}\s*$", re.IGNORECASE)


def _find_heading(lines: list[Line], pattern: re.Pattern, start: int, last_page: int):
    """Return (index, line_count) of the heading, trying one line and two joined lines."""
    for i in range(start, len(lines)):
        if lines[i].page_index > last_page:
            break
        if pattern.match(lines[i].text):
            return i, 1
        if i + 1 < len(lines) and pattern.match(f"{lines[i].text} {lines[i + 1].text}"):
            return i, 2
    return None


def _build(lines, headings, skip, warnings) -> list[Section]:
    """headings: list of (depth, title, line_index, heading_line_count) in order."""
    sections, stack = [], []
    for k, (depth, title, index, count) in enumerate(headings):
        end = headings[k + 1][2] if k + 1 < len(headings) else len(lines)
        stack = stack[:depth] + [title]
        if any(re.search(p, title) for p in skip):
            continue
        body = lines[index + count:end]
        if body:
            sections.append(Section(list(stack), body))
    if headings and headings[0][2] > 0:
        warnings.append(f"skipped {headings[0][2]} lines before the first heading (cover, contents)")
    return sections


def sections_from_outline(reader, lines, skip, warnings) -> list[Section]:
    headings, cursor = [], 0
    first_line_of_page = {}
    for i, line in enumerate(lines):
        first_line_of_page.setdefault(line.page_index, i)
    for depth, title, page in _outline(reader):
        start = max(cursor, first_line_of_page.get(page, cursor))
        found = _find_heading(lines, _title_pattern(title), start, page + 12)
        if found is None:
            warnings.append(f"outline heading not found in text, merged into previous section: {title!r}")
            continue
        index, count = found
        headings.append((depth, title, index, count))
        cursor = index + count
    return _build(lines, headings, skip, warnings)


def sections_from_patterns(lines, patterns, skip, warnings) -> list[Section]:
    """Fallback for PDFs without an outline: headings are whole lines matching a regex."""
    compiled = [(depth, re.compile(rf"^(?:{p})\s*$")) for depth, p in patterns]
    headings = []
    for i, line in enumerate(lines):
        for depth, pattern in compiled:
            if pattern.match(line.text):
                headings.append((depth, line.text, i, 1))
                break
    if not headings:
        warnings.append("no headings found; the whole document is one section")
        return [Section([], lines)]
    return _build(lines, headings, skip, warnings)


def read_sections(path: Path, skip=(), patterns=(), footnotes=False, warnings=None) -> list[Section]:
    warnings = [] if warnings is None else warnings
    reader, lines = read_lines(path, footnotes)
    if reader.outline:
        return sections_from_outline(reader, lines, skip, warnings)
    warnings.append("PDF has no outline; using heading patterns")
    return sections_from_patterns(lines, patterns, skip, warnings)
