"""Embed the chunks for dense retrieval.

    python -m ingest.embed [--strategy structured generic] [--batch-size 16] [--force]

For each strategy, reads data/processed/<strategy>/chunks.jsonl and writes
embeddings.npy (float32, one L2-normalised row per chunk, in file order) and
embeddings.json (model, prefix, chunk ids and a hash of the chunk file). A
strategy is skipped when its embeddings already match the current chunks;
--force recomputes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ingest.chunk import DOCUMENT_PREFIX, EMBEDDING_MODEL, OUT_DIR, STRATEGIES


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_current(folder: Path, chunk_hash: str) -> bool:
    meta_path, vectors_path = folder / "embeddings.json", folder / "embeddings.npy"
    if not (meta_path.exists() and vectors_path.exists()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return meta.get("chunks_sha256") == chunk_hash and meta.get("model") == EMBEDDING_MODEL


def embed_strategy(model, folder: Path, batch_size: int, force: bool) -> None:
    chunks_path = folder / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"{folder.name}: no chunks.jsonl, run `python -m ingest.chunk --strategy {folder.name}` first")
        return
    chunk_hash = sha256(chunks_path)
    if not force and is_current(folder, chunk_hash):
        print(f"{folder.name}: embeddings are up to date")
        return

    chunks = [json.loads(line) for line in chunks_path.open(encoding="utf-8")]
    texts = [DOCUMENT_PREFIX + c["text"] for c in chunks]
    print(f"{folder.name}: embedding {len(texts)} chunks ...")
    start = time.time()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    seconds = time.time() - start

    np.save(folder / "embeddings.npy", vectors)
    meta = {
        "model": EMBEDDING_MODEL,
        "document_prefix": DOCUMENT_PREFIX,
        "normalized": True,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "chunks_sha256": chunk_hash,
        "seconds": round(seconds, 1),
        "chunk_ids": [c["id"] for c in chunks],
    }
    (folder / "embeddings.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{folder.name}: {vectors.shape[0]} x {vectors.shape[1]} in {seconds:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strategy", nargs="+", choices=sorted(STRATEGIES), default=sorted(STRATEGIES))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="recompute even if embeddings are current")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    for strategy in args.strategy:
        embed_strategy(model, OUT_DIR / strategy, args.batch_size, args.force)


if __name__ == "__main__":
    main()
