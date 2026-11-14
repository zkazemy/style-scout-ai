"""Tests for catalog/embeddings.py.

Fast tests use a deterministic fake embedding function and a temp Chroma path
so they run fully offline without downloading the ONNX model.

Tests that require the real model are marked @pytest.mark.slow and excluded
from the default test run (configure with: pytest -m "not slow").
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from chromadb import Documents, Embeddings

from stylescout.catalog.embeddings import (
    METADATA_FIELDS,
    build_index,
    build_product_text,
    query_index,
)

# ---------------------------------------------------------------------------
# Fake embedding function (offline, deterministic)
# ---------------------------------------------------------------------------


class FakeEmbeddingFn:
    """Deterministic 8-dim fake embeddings based on string hash.

    Two identical strings always produce the same embedding vector.
    Satisfies the EmbeddingFn protocol without loading any model.

    The name/build_from_config/get_config/is_legacy methods are required by
    chromadb 1.5.x when passing a custom embedding function to a collection.
    """

    def __init__(self) -> None:
        pass

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        result = []
        for text in input:
            digest = hashlib.md5(text.encode()).digest()
            vec = [((b / 255.0) * 2 - 1) for b in digest[:8]]
            result.append(vec)
        return result

    @staticmethod
    def name() -> str:
        return "fake"

    @staticmethod
    def build_from_config(config: dict) -> FakeEmbeddingFn:
        return FakeEmbeddingFn()

    def get_config(self) -> dict:
        return {}

    def embed_query(self, input: Documents) -> Embeddings:  # noqa: A002
        return self(input)

    def is_legacy(self) -> bool:
        return False

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": f"PROD-{i:05d}",
        "name": f"Product {i}",
        "category": ["shirt", "dress", "shoes", "trousers", "jacket", "accessory"][i % 6],
        "color": ["black", "blue", "red", "white", "green", "grey"][i % 6],
        "occasion": ["casual", "formal", "sport", "evening", "casual", "formal"][i % 6],
        "description": f"A fine item number {i} suitable for many occasions.",
        "price_eur": 20.0 + i * 5,
        "in_stock": i % 3,
    }
    for i in range(10)
]


@pytest.fixture()
def chroma_dir(tmp_path: Path) -> Path:
    return tmp_path / "chroma"


@pytest.fixture()
def fake_ef() -> FakeEmbeddingFn:
    return FakeEmbeddingFn()


@pytest.fixture()
def built_index(chroma_dir: Path, fake_ef: FakeEmbeddingFn, tmp_path: Path) -> Path:
    """Build a small index from the fixture products and return chroma_dir."""
    import sqlite3

    # Write a tiny SQLite db with the fixture products
    db_path = tmp_path / "fixture.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE products (
                product_id TEXT PRIMARY KEY, name TEXT, category TEXT, color TEXT,
                occasion TEXT, description TEXT, price_eur REAL, in_stock INTEGER,
                brand TEXT, sizes TEXT
            )
        """)
        conn.executemany(
            "INSERT INTO products VALUES (:product_id,:name,:category,:color,:occasion,"
            ":description,:price_eur,:in_stock,'BrandX','{}')",
            FIXTURE_PRODUCTS,
        )

    build_index(
        db_path=db_path,
        chroma_path=chroma_dir,
        collection_name="test_products",
        embed_fn=fake_ef,
    )
    return chroma_dir


# ---------------------------------------------------------------------------
# Tests: build_product_text
# ---------------------------------------------------------------------------


def test_build_product_text_contains_all_fields() -> None:
    product = {
        "name": "Cool Shirt",
        "category": "shirt",
        "color": "blue",
        "occasion": "casual",
        "description": "A relaxed fit shirt.",
    }
    text = build_product_text(product)
    assert "Cool Shirt" in text
    assert "shirt" in text
    assert "blue" in text
    assert "casual" in text
    assert "A relaxed fit shirt." in text


def test_build_product_text_order() -> None:
    product = {
        "name": "N",
        "category": "C",
        "color": "CO",
        "occasion": "OC",
        "description": "D",
    }
    assert build_product_text(product) == "N C CO OC D"


def test_build_product_text_missing_field_skipped() -> None:
    product = {"name": "Jacket", "category": "jacket"}
    text = build_product_text(product)
    assert "Jacket" in text
    assert "jacket" in text
    assert "  " not in text  # no double spaces from empty fields


# ---------------------------------------------------------------------------
# Tests: build_index
# ---------------------------------------------------------------------------


def test_build_index_count(built_index: Path, fake_ef: FakeEmbeddingFn) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(built_index))
    col = client.get_collection("test_products", embedding_function=fake_ef)
    assert col.count() == len(FIXTURE_PRODUCTS)


def test_build_index_idempotent(
    built_index: Path, fake_ef: FakeEmbeddingFn, tmp_path: Path
) -> None:
    """Building the index twice must leave the same document count."""
    db_path = tmp_path / "fixture.db"  # reuse the already-created db

    # Build a second time
    build_index(
        db_path=db_path,
        chroma_path=built_index,
        collection_name="test_products",
        embed_fn=fake_ef,
    )

    import chromadb

    client = chromadb.PersistentClient(path=str(built_index))
    col = client.get_collection("test_products", embedding_function=fake_ef)
    assert col.count() == len(FIXTURE_PRODUCTS)


def test_build_index_metadata_stored(built_index: Path, fake_ef: FakeEmbeddingFn) -> None:
    import chromadb

    client = chromadb.PersistentClient(path=str(built_index))
    col = client.get_collection("test_products", embedding_function=fake_ef)
    result = col.get(ids=["PROD-00000"], include=["metadatas"])
    meta = result["metadatas"][0]
    for field in METADATA_FIELDS:
        assert field in meta, f"metadata field '{field}' missing"


# ---------------------------------------------------------------------------
# Tests: query_index
# ---------------------------------------------------------------------------


def test_query_returns_known_ids(built_index: Path, fake_ef: FakeEmbeddingFn) -> None:
    known_ids = {p["product_id"] for p in FIXTURE_PRODUCTS}
    results = query_index(
        query="casual shirt",
        chroma_path=built_index,
        collection_name="test_products",
        n_results=5,
        embed_fn=fake_ef,
    )
    assert len(results) == 5
    for pid, dist in results:
        assert pid in known_ids
        assert isinstance(dist, float)


def test_query_with_where_filter(built_index: Path, fake_ef: FakeEmbeddingFn) -> None:
    results = query_index(
        query="formal occasion",
        chroma_path=built_index,
        collection_name="test_products",
        n_results=3,
        where={"category": "shirt"},
        embed_fn=fake_ef,
    )
    # All returned ids must be shirts
    import chromadb

    client = chromadb.PersistentClient(path=str(built_index))
    col = client.get_collection("test_products", embedding_function=fake_ef)
    for pid, _ in results:
        meta = col.get(ids=[pid], include=["metadatas"])["metadatas"][0]
        assert meta["category"] == "shirt"


# ---------------------------------------------------------------------------
# Slow tests (require real ONNX model)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_real_model_query(tmp_path: Path) -> None:
    """Smoke test: real model embeds a string without error."""
    from stylescout.catalog.embeddings import make_embedding_fn

    ef = make_embedding_fn()
    embeddings = ef(["black sneakers for casual wear"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0
