"""Hybrid retrieval over the chunks: BM25 keyword search plus dense embeddings.

The two rankings are combined with reciprocal rank fusion (RRF): each chunk
scores sum(1 / (k + rank)) over the rankings it appears in. RRF uses only
ranks, so BM25 scores and cosine similarities never need to be put on the
same scale.

In hybrid mode a third ranking can be fused in: exact identifier lookup.
If the question names an ID such as AML.T0051 or GOVERN 1.1, the chunks
whose own ID it is rank first. Without it, BM25 favours the many short
chunks that merely mention the ID (for example "mitigation X applied to
AML.T0051") over the chunk that defines it.

`mode` selects "hybrid", "bm25" or "dense", and `id_lookup` can be turned
off, so each part can be evaluated alone.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from ingest.chunk import EMBEDDING_MODEL, ID_PATTERNS, OUT_DIR, QUERY_PREFIX, find_ids

MODES = ("hybrid", "bm25", "dense")
RRF_K = 60
CANDIDATES = 50  # how deep each ranking goes before fusion

# Keeps framework identifiers whole: "gv-1.1-001", "aml.t0051.001", "llm01", "1.1".
TOKEN = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOPWORDS = set("""
a an and are as at be by can do does for from has have how i in is it its
of on or should that the their this to was what when where which who why
will with
""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class Hit:
    chunk: dict
    score: float
    bm25_rank: int | None
    dense_rank: int | None


class Retriever:
    def __init__(self, strategy: str = "structured", folder: Path | None = None):
        self.folder = folder or OUT_DIR / strategy
        chunks_path = self.folder / "chunks.jsonl"
        if not chunks_path.exists():
            raise FileNotFoundError(f"{chunks_path} not found; run `python -m ingest.chunk --strategy {strategy}`")
        self.chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8")]
        self._chunks_sha = hashlib.sha256(chunks_path.read_bytes()).hexdigest()
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.by_id = {}
        for i, c in enumerate(self.chunks):
            if c["ids"] and c["chunk_type"] != "window":
                self.by_id.setdefault(c["ids"][0], []).append(i)

    @cached_property
    def vectors(self) -> np.ndarray:
        meta = json.loads((self.folder / "embeddings.json").read_text(encoding="utf-8"))
        if meta["chunks_sha256"] != self._chunks_sha or meta["model"] != EMBEDDING_MODEL:
            raise RuntimeError(f"embeddings in {self.folder} are stale; run `python -m ingest.embed`")
        return np.load(self.folder / "embeddings.npy")

    @cached_property
    def encoder(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)

    def bm25_ranking(self, query: str) -> list[int]:
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        order = np.argsort(-scores)[:CANDIDATES]
        return [int(i) for i in order if scores[i] > 0]

    def dense_ranking(self, query: str) -> list[int]:
        q = self.encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        return [int(i) for i in np.argsort(-(self.vectors @ q))[:CANDIDATES]]

    def id_ranking(self, query: str) -> list[int]:
        """Chunks whose own identifier is named in the query, in the query's order."""
        ranking = []
        for ident in find_ids(query.upper(), ID_PATTERNS):
            ranking += [i for i in self.by_id.get(ident, []) if i not in ranking]
        return ranking

    def search(self, query: str, k: int = 8, mode: str = "hybrid", id_lookup: bool = True) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        bm25 = self.bm25_ranking(query) if mode in ("hybrid", "bm25") else []
        dense = self.dense_ranking(query) if mode in ("hybrid", "dense") else []
        ids = self.id_ranking(query) if mode == "hybrid" and id_lookup else []
        bm25_rank = {i: r for r, i in enumerate(bm25, start=1)}
        dense_rank = {i: r for r, i in enumerate(dense, start=1)}
        id_rank = {i: r for r, i in enumerate(ids, start=1)}
        scores = {}
        for ranks in (bm25_rank, dense_rank, id_rank):
            for i, r in ranks.items():
                scores[i] = scores.get(i, 0.0) + 1.0 / (RRF_K + r)
        top = sorted(scores, key=lambda i: -scores[i])[:k]
        return [Hit(self.chunks[i], scores[i], bm25_rank.get(i), dense_rank.get(i)) for i in top]
