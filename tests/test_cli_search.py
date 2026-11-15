"""Tests for the `search` CLI command.

All tests use a small SQLite fixture and a fake embedding function so they run
fully offline without downloading the ONNX model.

The fixture products are the same five items used in test_hybrid.py.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
import pytest

from stylescout.catalog.embeddings import EmbeddingFn
from stylescout.catalog.search import SearchFilters, build_fts_index
from stylescout.cli import (
    format_table,
    parse_filters,
    run_hybrid,
    run_keyword,
    run_vector,
    search_cmd,
)

# ---------------------------------------------------------------------------
# Fake embedding function (identical to test_hybrid.py)
# ---------------------------------------------------------------------------

_UNIT_VEC = [1.0, 0.0, 0.0, 0.0]


class _FakeEmbedFn:
    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [_UNIT_VEC for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [_UNIT_VEC for _ in input]

    def name(self) -> str:  # noqa: A003
        return "default"


FAKE_EMBED_FN: EmbeddingFn = _FakeEmbedFn()

# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

FIXTURE_PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": "PROD-00001",
        "name": "Blue Floral Dress",
        "category": "dress",
        "color": "blue",
        "brand": "Zara",
        "occasion": "casual",
        "description": "A light summer item with floral print.",
        "price_eur": 49.99,
        "sizes": json.dumps({"XS": 0, "S": 3, "M": 5, "L": 2, "XL": 0}),
        "in_stock": 1,
    },
    {
        "product_id": "PROD-00002",
        "name": "Black Leather Jacket",
        "category": "jacket",
        "color": "black",
        "brand": "Mango",
        "occasion": "casual",
        "description": "Classic black leather biker item.",
        "price_eur": 129.99,
        "sizes": json.dumps({"XS": 0, "S": 0, "M": 0, "L": 0, "XL": 0}),
        "in_stock": 0,
    },
    {
        "product_id": "PROD-00003",
        "name": "White Formal Shirt",
        "category": "shirt",
        "color": "white",
        "brand": "HM",
        "occasion": "formal",
        "description": "Crisp white button-down for office wear.",
        "price_eur": 39.99,
        "sizes": json.dumps({"XS": 1, "S": 4, "M": 0, "L": 3, "XL": 2, "XXL": 1}),
        "in_stock": 1,
    },
    {
        "product_id": "PROD-00004",
        "name": "Red Running Shoes",
        "category": "shoes",
        "color": "red",
        "brand": "Nike",
        "occasion": "sport",
        "description": "Lightweight running item with cushioned sole.",
        "price_eur": 89.99,
        "sizes": json.dumps({"38": 2, "39": 0, "40": 5, "41": 3}),
        "in_stock": 1,
    },
    {
        "product_id": "PROD-00005",
        "name": "Navy Chino Trousers",
        "category": "trousers",
        "color": "navy",
        "brand": "Gap",
        "occasion": "casual",
        "description": "Slim-fit navy chinos for smart wear.",
        "price_eur": 59.99,
        "sizes": json.dumps({"30": 0, "32": 0, "34": 0}),
        "in_stock": 0,
    },
]

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    product_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    color       TEXT NOT NULL,
    brand       TEXT NOT NULL,
    occasion    TEXT,
    description TEXT,
    price_eur   REAL NOT NULL,
    sizes       TEXT NOT NULL,
    in_stock    INTEGER NOT NULL
)
"""

_INSERT_SQL = """
INSERT OR REPLACE INTO products
    (product_id, name, category, color, brand, occasion, description,
     price_eur, sizes, in_stock)
VALUES
    (:product_id, :name, :category, :color, :brand, :occasion, :description,
     :price_eur, :sizes, :in_stock)
"""


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "test_catalog.db"
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.executemany(_INSERT_SQL, FIXTURE_PRODUCTS)
        conn.commit()
    build_fts_index(path)
    return path


@pytest.fixture()
def chroma_path(tmp_path: Path, db_path: Path) -> Path:
    from stylescout.catalog.embeddings import build_product_text

    path = tmp_path / "chroma"
    path.mkdir()
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name="products",
        metadata={"hnsw:space": "cosine"},
        embedding_function=FAKE_EMBED_FN,  # type: ignore[arg-type]
    )
    ids = [p["product_id"] for p in FIXTURE_PRODUCTS]
    docs = [build_product_text(p) for p in FIXTURE_PRODUCTS]
    metadatas = [
        {
            "category": p["category"],
            "color": p["color"],
            "occasion": p["occasion"],
            "price_eur": p["price_eur"],
            "in_stock": p["in_stock"],
        }
        for p in FIXTURE_PRODUCTS
    ]
    collection.upsert(ids=ids, documents=docs, metadatas=metadatas)
    return path


def _make_args(**kwargs: Any) -> argparse.Namespace:
    """Build a Namespace with search defaults, overriding with kwargs."""
    defaults = dict(
        query="dress",
        mode="hybrid",
        top_k=5,
        category=None,
        color=None,
        occasion=None,
        brand=None,
        min_price=None,
        max_price=None,
        size=None,
        include_out_of_stock=False,
        db="data/catalog.db",
        chroma="data/chroma",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# parse_filters
# ---------------------------------------------------------------------------


def test_parse_filters_defaults_to_in_stock_only() -> None:
    args = _make_args()
    f = parse_filters(args)
    assert f.in_stock_only is True
    assert f.category is None


def test_parse_filters_include_out_of_stock() -> None:
    args = _make_args(include_out_of_stock=True)
    f = parse_filters(args)
    assert f.in_stock_only is False


def test_parse_filters_all_fields() -> None:
    args = _make_args(
        category="dress",
        color="blue",
        occasion="casual",
        brand="Zara",
        min_price=10.0,
        max_price=100.0,
        size="M",
        include_out_of_stock=True,
    )
    f = parse_filters(args)
    assert f == SearchFilters(
        category="dress",
        color="blue",
        occasion="casual",
        brand="Zara",
        min_price=10.0,
        max_price=100.0,
        size="M",
        in_stock_only=False,
    )


# ---------------------------------------------------------------------------
# run_keyword
# ---------------------------------------------------------------------------


def test_run_keyword_returns_results(db_path: Path) -> None:
    results = run_keyword("dress", SearchFilters(in_stock_only=False), top_k=5, db_path=db_path)
    assert len(results) > 0
    assert results[0]["name"] == "Blue Floral Dress"


def test_run_keyword_has_bm25_score(db_path: Path) -> None:
    results = run_keyword("dress", SearchFilters(in_stock_only=False), top_k=5, db_path=db_path)
    assert all("bm25_score" in r for r in results)


def test_run_keyword_applies_in_stock_filter(db_path: Path) -> None:
    # Use a broad query so all items are candidates; filter then removes out-of-stock ones.
    in_stock_only = run_keyword(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=True),
        top_k=10,
        db_path=db_path,
    )
    all_stock = run_keyword(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=False),
        top_k=10,
        db_path=db_path,
    )
    ids_in_stock = {r["product_id"] for r in in_stock_only}
    assert "PROD-00002" not in ids_in_stock  # Black Leather Jacket is out of stock
    assert "PROD-00002" in {r["product_id"] for r in all_stock}


def test_run_keyword_applies_color_filter(db_path: Path) -> None:
    results = run_keyword(
        "dress shirt jacket shoes trousers",
        SearchFilters(color="blue", in_stock_only=False),
        top_k=10,
        db_path=db_path,
    )
    assert all(r["color"] == "blue" for r in results)


def test_run_keyword_ranks_are_sequential(db_path: Path) -> None:
    results = run_keyword(
        "dress shirt", SearchFilters(in_stock_only=False), top_k=5, db_path=db_path
    )
    assert [r["rank"] for r in results] == list(range(1, len(results) + 1))


# ---------------------------------------------------------------------------
# run_vector
# ---------------------------------------------------------------------------


def test_run_vector_returns_results(db_path: Path, chroma_path: Path) -> None:
    results = run_vector(
        "dress",
        SearchFilters(in_stock_only=False),
        top_k=5,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) > 0


def test_run_vector_has_distance(db_path: Path, chroma_path: Path) -> None:
    results = run_vector(
        "dress",
        SearchFilters(in_stock_only=False),
        top_k=5,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert all("distance" in r for r in results)


def test_run_vector_applies_in_stock_filter(db_path: Path, chroma_path: Path) -> None:
    results = run_vector(
        "dress shirt jacket",
        SearchFilters(in_stock_only=True),
        top_k=10,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids = {r["product_id"] for r in results}
    assert "PROD-00002" not in ids  # out of stock
    assert "PROD-00005" not in ids  # out of stock


def test_run_vector_applies_size_filter(db_path: Path, chroma_path: Path) -> None:
    """Size filter must exclude PROD-00003 (White Formal Shirt, M=0)."""
    results = run_vector(
        "dress shirt",
        SearchFilters(size="M", in_stock_only=False),
        top_k=10,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids = {r["product_id"] for r in results}
    assert "PROD-00001" in ids  # Blue Floral Dress M=5 → passes
    assert "PROD-00003" not in ids  # White Formal Shirt M=0 → excluded


# ---------------------------------------------------------------------------
# run_hybrid
# ---------------------------------------------------------------------------


def test_run_hybrid_returns_results(db_path: Path, chroma_path: Path) -> None:
    results = run_hybrid(
        "dress",
        SearchFilters(in_stock_only=False),
        top_k=5,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) > 0


def test_run_hybrid_has_rank_columns(db_path: Path, chroma_path: Path) -> None:
    results = run_hybrid(
        "dress",
        SearchFilters(in_stock_only=False),
        top_k=5,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert all("keyword_rank" in r and "vector_rank" in r for r in results)


def test_run_hybrid_include_out_of_stock(db_path: Path, chroma_path: Path) -> None:
    """--include-out-of-stock must broaden results."""
    in_stock = run_hybrid(
        "shirt jacket dress trousers shoes",
        SearchFilters(in_stock_only=True),
        top_k=10,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    all_stock = run_hybrid(
        "shirt jacket dress trousers shoes",
        SearchFilters(in_stock_only=False),
        top_k=10,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids_in_stock = {r["product_id"] for r in in_stock}
    ids_all = {r["product_id"] for r in all_stock}
    # PROD-00002 and PROD-00005 are out of stock
    assert "PROD-00002" not in ids_in_stock
    assert "PROD-00002" in ids_all


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_hybrid_columns() -> None:
    rows = [
        {
            "rank": 1,
            "product_id": "P1",
            "name": "Test Dress",
            "category": "dress",
            "color": "blue",
            "price_eur": 49.99,
            "rrf_score": 0.03,
            "keyword_rank": 1,
            "vector_rank": 2,
        }
    ]
    table = format_table(rows, "hybrid")
    assert "KW" in table
    assert "Vec" in table
    assert "Test Dress" in table
    assert "1" in table  # keyword_rank


def test_format_table_vector_columns() -> None:
    rows = [
        {
            "rank": 1,
            "product_id": "P1",
            "name": "Test Dress",
            "category": "dress",
            "color": "blue",
            "price_eur": 49.99,
            "distance": 0.242,
        }
    ]
    table = format_table(rows, "vector")
    assert "Dist" in table
    assert "0.242" in table
    assert "KW" not in table


def test_format_table_keyword_columns() -> None:
    rows = [
        {
            "rank": 1,
            "product_id": "P1",
            "name": "Test Dress",
            "category": "dress",
            "color": "blue",
            "price_eur": 49.99,
            "bm25_score": -3.5,
        }
    ]
    table = format_table(rows, "keyword")
    assert "BM25" in table
    assert "Dist" not in table
    assert "KW" not in table


def test_format_table_hybrid_none_rank_shown_as_dash() -> None:
    rows = [
        {
            "rank": 1,
            "product_id": "P1",
            "name": "Test Item",
            "category": "shoes",
            "color": "black",
            "price_eur": 80.0,
            "rrf_score": 0.016,
            "keyword_rank": None,
            "vector_rank": 3,
        }
    ]
    table = format_table(rows, "hybrid")
    assert "-" in table


def test_format_table_empty_returns_empty_string() -> None:
    assert format_table([], "hybrid") == ""
    assert format_table([], "keyword") == ""
    assert format_table([], "vector") == ""


# ---------------------------------------------------------------------------
# search_cmd — missing catalog / index errors
# ---------------------------------------------------------------------------


def test_search_cmd_missing_catalog_prints_message_and_exits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(query="dress", mode="keyword", db=str(tmp_path / "no.db"))
    with pytest.raises(SystemExit) as exc:
        search_cmd(args)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "build-catalog" in captured.err


def test_search_cmd_missing_vector_index_prints_message_and_exits(
    db_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    empty_chroma = tmp_path / "empty_chroma"
    empty_chroma.mkdir()
    args = _make_args(
        query="dress",
        mode="vector",
        db=str(db_path),
        chroma=str(empty_chroma),
    )
    with pytest.raises(SystemExit) as exc:
        search_cmd(args, embed_fn=FAKE_EMBED_FN)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "build-index" in captured.err


# ---------------------------------------------------------------------------
# search_cmd — no-results message
# ---------------------------------------------------------------------------


def test_search_cmd_no_results_prints_hint(
    db_path: Path, chroma_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(
        query="dress",
        mode="keyword",
        color="purple",  # no purple item in fixture
        db=str(db_path),
        chroma=str(chroma_path),
        include_out_of_stock=True,
    )
    search_cmd(args, embed_fn=FAKE_EMBED_FN)
    captured = capsys.readouterr()
    assert "No products matched" in captured.out


# ---------------------------------------------------------------------------
# search_cmd — end-to-end per mode
# ---------------------------------------------------------------------------


def test_search_cmd_keyword_produces_table(
    db_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(
        query="dress",
        mode="keyword",
        include_out_of_stock=True,
        db=str(db_path),
    )
    search_cmd(args)
    captured = capsys.readouterr()
    assert "Blue Floral Dress" in captured.out
    assert "BM25" in captured.out


def test_search_cmd_vector_produces_table(
    db_path: Path, chroma_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(
        query="dress",
        mode="vector",
        include_out_of_stock=True,
        db=str(db_path),
        chroma=str(chroma_path),
    )
    search_cmd(args, embed_fn=FAKE_EMBED_FN)
    captured = capsys.readouterr()
    assert "Dist" in captured.out


def test_search_cmd_hybrid_produces_table(
    db_path: Path, chroma_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = _make_args(
        query="dress",
        mode="hybrid",
        include_out_of_stock=True,
        db=str(db_path),
        chroma=str(chroma_path),
    )
    search_cmd(args, embed_fn=FAKE_EMBED_FN)
    captured = capsys.readouterr()
    assert "KW" in captured.out
    assert "Vec" in captured.out


def test_search_cmd_include_out_of_stock_broadens_results(
    db_path: Path, chroma_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args_in = _make_args(
        query="shirt jacket dress trousers shoes",
        mode="hybrid",
        include_out_of_stock=False,
        db=str(db_path),
        chroma=str(chroma_path),
        top_k=10,
    )
    args_all = _make_args(
        query="shirt jacket dress trousers shoes",
        mode="hybrid",
        include_out_of_stock=True,
        db=str(db_path),
        chroma=str(chroma_path),
        top_k=10,
    )
    search_cmd(args_in, embed_fn=FAKE_EMBED_FN)
    out_in = capsys.readouterr().out

    search_cmd(args_all, embed_fn=FAKE_EMBED_FN)
    out_all = capsys.readouterr().out

    # Black Leather Jacket (out of stock) must appear only in the inclusive run
    assert "Black Leather Jacket" not in out_in
    assert "Black Leather Jacket" in out_all
