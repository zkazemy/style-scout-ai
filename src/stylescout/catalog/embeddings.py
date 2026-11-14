"""Build and query the Chroma vector index for the product catalog.

Embedding model: all-MiniLM-L6-v2 via chromadb's built-in ONNXMiniLM_L6_V2.
No PyTorch or sentence-transformers required; depends only on onnxruntime and
tokenizers, both already in pyproject.toml as transitive chromadb dependencies.

The embedding function is injectable so tests can pass a fast fake and run
fully offline without downloading the ONNX model.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb
from chromadb import Documents, Embeddings

logger = logging.getLogger(__name__)

COLLECTION_NAME = "products"
BATCH_SIZE = 100  # upsert in batches to avoid large in-memory payloads

# Metadata fields stored per document so Chroma where-filters work on them.
METADATA_FIELDS = ("category", "color", "occasion", "price_eur", "in_stock")


# ---------------------------------------------------------------------------
# Embedding function protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EmbeddingFn(Protocol):
    """Minimal protocol for an injectable embedding function.

    Matches chromadb's EmbeddingFunction.__call__ signature so both the real
    ONNXMiniLM_L6_V2 and any test double satisfy it without inheritance.
    """

    def __call__(self, input: Documents) -> Embeddings: ...  # noqa: A002


def make_embedding_fn() -> EmbeddingFn:
    """Return the default embedding function (all-MiniLM-L6-v2 via ONNX).

    Downloads ~79 MB to ~/.cache/chroma/onnx_models/ on first call.
    Call this once and reuse the instance across build_index and query_index
    so the model is loaded only once per process.
    """
    from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2()


# ---------------------------------------------------------------------------
# Text representation
# ---------------------------------------------------------------------------


def build_product_text(product: dict[str, Any]) -> str:
    """Return the text string embedded for a product document.

    Format (space-separated, in this order):
        "{name} {category} {color} {occasion} {description}"

    Rationale: name + category + color + occasion give the structured signal
    the retrieval spec (§5.1) calls out; description adds free-text context.
    All fields are expected to be non-empty strings after the loader's clean()
    step; missing values fall back to an empty string.
    """
    fields = (
        product.get("name", ""),
        product.get("category", ""),
        product.get("color", ""),
        product.get("occasion", ""),
        product.get("description", ""),
    )
    return " ".join(f for f in fields if f)


# ---------------------------------------------------------------------------
# Index build
# ---------------------------------------------------------------------------


def build_index(
    db_path: str | Path = "data/catalog.db",
    chroma_path: str | Path = "data/chroma",
    collection_name: str = COLLECTION_NAME,
    embed_fn: EmbeddingFn | None = None,
) -> dict[str, int]:
    """Read products from SQLite, embed them, and upsert into Chroma.

    Idempotent: the collection is deleted and recreated on every call so no
    stale documents survive a catalog rebuild. Upsert (not add) is used within
    batches so interrupted runs can be retried safely.

    Returns a summary dict with key "indexed" (number of rows written).
    """
    if embed_fn is None:
        embed_fn = make_embedding_fn()

    db_path = Path(db_path)
    chroma_path = Path(chroma_path)
    chroma_path.mkdir(parents=True, exist_ok=True)

    # Load products from SQLite
    products = _load_products(db_path)
    logger.info("build_index: loaded %d products from %s", len(products), db_path)

    client = chromadb.PersistentClient(path=str(chroma_path))

    # Delete + recreate guarantees no stale vectors survive a rebuild
    try:
        client.delete_collection(collection_name)
        logger.info("build_index: deleted existing collection '%s'", collection_name)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=embed_fn,  # type: ignore[arg-type]
    )

    # Upsert in batches
    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start : start + BATCH_SIZE]
        ids = [p["product_id"] for p in batch]
        documents = [build_product_text(p) for p in batch]
        metadatas = [{field: p.get(field, "") for field in METADATA_FIELDS} for p in batch]
        collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    count = collection.count()
    logger.info("build_index: collection '%s' has %d documents", collection_name, count)
    return {"indexed": count}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def query_index(
    query: str,
    chroma_path: str | Path = "data/chroma",
    collection_name: str = COLLECTION_NAME,
    n_results: int = 10,
    where: dict[str, Any] | None = None,
    embed_fn: EmbeddingFn | None = None,
) -> list[tuple[str, float]]:
    """Query the vector index and return (product_id, distance) pairs.

    Uses the same embedding function as build_index (pass the same instance
    for efficiency, or omit to create a fresh one). Lower cosine distance
    means more similar.

    Args:
        query: Free-text query string.
        where: Optional Chroma metadata filter, e.g. {"category": "shoes"}.
        n_results: Maximum number of results to return.
    """
    if embed_fn is None:
        embed_fn = make_embedding_fn()

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embed_fn,  # type: ignore[arg-type]
    )

    kwargs: dict[str, Any] = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    result = collection.query(**kwargs)

    ids: list[str] = result["ids"][0]
    distances: list[float] = result["distances"][0]
    return list(zip(ids, distances))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_products(db_path: Path) -> list[dict[str, Any]]:
    """Fetch all rows from the products table as plain dicts."""
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM products").fetchall()
    return [dict(row) for row in rows]
