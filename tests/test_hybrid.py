"""Tests for catalog/hybrid.py.

All tests use either:
- Pure in-memory computation (rrf_fuse tests), or
- A small SQLite fixture + a deterministic fake embedding function so tests
  run fully offline without downloading any model.

The fake embedding function returns a fixed unit-vector for every document,
which makes every cosine distance identical.  That means vector ranking is
arbitrary (Chroma's insertion order), which is fine for testing the pipeline
plumbing; real ranking quality is measured in retrieval evaluation (T10).
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

import chromadb
import pytest

from stylescout.catalog.embeddings import EmbeddingFn
from stylescout.catalog.hybrid import (
    HybridHit,
    _confirm_with_sqlite,
    _hybrid_search_sync,
    rrf_fuse,
)
from stylescout.catalog.search import SearchFilters, build_fts_index

# ---------------------------------------------------------------------------
# Fake embedding function
# ---------------------------------------------------------------------------

# 4-dimensional unit vector; all documents get the same embedding so distance
# is identical across them (Chroma still returns results, in insertion order).
_UNIT_VEC = [1.0, 0.0, 0.0, 0.0]


class _FakeEmbedFn:
    """Returns identical 4-d unit vectors so tests run fully offline.

    name() returns "default" so Chroma's embedding-function conflict check
    passes without requiring us to match a specific registered name.

    embed_query() is required by newer Chroma versions, which call it at
    query time (vs __call__ at upsert time).
    """

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [_UNIT_VEC for _ in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return [_UNIT_VEC for _ in input]

    def name(self) -> str:  # noqa: A003
        return "default"


FAKE_EMBED_FN: EmbeddingFn = _FakeEmbedFn()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURE_PRODUCTS: list[dict[str, Any]] = [
    # category / occasion words intentionally NOT in name/description so the
    # FTS-column tests in test_search.py stay valid here too.
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
    """Build a minimal Chroma collection from the fixture products."""
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


# ---------------------------------------------------------------------------
# rrf_fuse — pure unit tests
# ---------------------------------------------------------------------------


def test_rrf_item_in_both_lists_beats_item_in_one() -> None:
    """An id ranked 1st in both lists must outscore an id ranked 1st in one."""
    # "shared" appears in both; "solo" appears only in list A
    fused = rrf_fuse(
        rankings=[["shared", "solo"], ["shared"]],
        weights=[1.0, 1.0],
        k=60,
    )
    scores = dict(fused)
    assert scores["shared"] > scores["solo"]


def test_rrf_weights_change_order() -> None:
    """Increasing a list's weight makes its top item beat the other list's top."""
    # "kw_top" is rank-1 in list 0; "vec_top" is rank-1 in list 1
    # With equal weights they tie at rank-1 score; with kw_weight=2 kw_top wins
    fused_kw_heavy = rrf_fuse(
        rankings=[["kw_top"], ["vec_top"]],
        weights=[2.0, 1.0],
        k=60,
    )
    order = [pid for pid, _ in fused_kw_heavy]
    assert order[0] == "kw_top"

    fused_vec_heavy = rrf_fuse(
        rankings=[["kw_top"], ["vec_top"]],
        weights=[1.0, 2.0],
        k=60,
    )
    order = [pid for pid, _ in fused_vec_heavy]
    assert order[0] == "vec_top"


def test_rrf_k_changes_scores() -> None:
    """Smaller k concentrates weight on top ranks; larger k flattens the curve."""
    fused_small_k = dict(rrf_fuse([["A", "B"]], [1.0], k=1))
    fused_large_k = dict(rrf_fuse([["A", "B"]], [1.0], k=1000))

    # With k=1: score(rank1)=1/2, score(rank2)=1/3 → ratio 1.5
    assert math.isclose(fused_small_k["A"] / fused_small_k["B"], 1.5, rel_tol=1e-9)
    # With k=1000: score(rank1)=1/1001, score(rank2)=1/1002 → ratio ~1.001
    ratio_large = fused_large_k["A"] / fused_large_k["B"]
    assert ratio_large < 1.01


def test_rrf_tie_breaking_is_deterministic() -> None:
    """Items with equal score are ordered by product_id ascending."""
    # Two items, each rank-1 in separate equal-weight lists → same score
    fused = rrf_fuse(
        rankings=[["PROD-B"], ["PROD-A"]],
        weights=[1.0, 1.0],
        k=60,
    )
    assert fused[0][0] == "PROD-A"
    assert fused[1][0] == "PROD-B"


def test_rrf_empty_ranking_list() -> None:
    """An empty inner list contributes nothing; the other list wins."""
    fused = rrf_fuse(rankings=[[], ["only"]], weights=[1.0, 1.0], k=60)
    assert len(fused) == 1
    assert fused[0][0] == "only"


def test_rrf_both_empty_returns_empty() -> None:
    assert rrf_fuse(rankings=[[], []], weights=[1.0, 1.0], k=60) == []


def test_rrf_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError, match="same length"):
        rrf_fuse(rankings=[["A"]], weights=[1.0, 1.0], k=60)


def test_rrf_scores_are_positive() -> None:
    fused = rrf_fuse(rankings=[["A", "B", "C"]], weights=[1.0], k=60)
    assert all(score > 0 for _, score in fused)


# ---------------------------------------------------------------------------
# _confirm_with_sqlite — unit tests
# ---------------------------------------------------------------------------


def test_confirm_with_sqlite_preserves_rrf_order(db_path: Path) -> None:
    """Rows are returned in candidate_ids order, not SQLite row order."""
    # Feed ids in reverse insertion order; output must respect that order
    ids = ["PROD-00004", "PROD-00001"]
    rows = _confirm_with_sqlite(ids, SearchFilters(in_stock_only=False), db_path)
    assert [r["product_id"] for r in rows] == ids


def test_confirm_with_sqlite_filters_out_non_matching(db_path: Path) -> None:
    ids = ["PROD-00001", "PROD-00002", "PROD-00003", "PROD-00004"]
    rows = _confirm_with_sqlite(ids, SearchFilters(in_stock_only=True), db_path)
    returned_ids = {r["product_id"] for r in rows}
    assert "PROD-00002" not in returned_ids  # out of stock


def test_confirm_with_sqlite_size_filter(db_path: Path) -> None:
    """Size filter must be enforced: PROD-00001 has M=5, PROD-00003 has M=0."""
    ids = ["PROD-00001", "PROD-00003"]
    rows = _confirm_with_sqlite(ids, SearchFilters(size="M", in_stock_only=False), db_path)
    returned_ids = {r["product_id"] for r in rows}
    assert "PROD-00001" in returned_ids
    assert "PROD-00003" not in returned_ids


def test_confirm_with_sqlite_empty_candidates(db_path: Path) -> None:
    assert _confirm_with_sqlite([], SearchFilters(), db_path) == []


# ---------------------------------------------------------------------------
# _hybrid_search_sync — integration tests on the fixture
# ---------------------------------------------------------------------------


def test_hybrid_returns_hybrid_hit_instances(db_path: Path, chroma_path: Path) -> None:
    results = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) > 0
    assert all(isinstance(r, HybridHit) for r in results)


def test_hybrid_top_k_respected(db_path: Path, chroma_path: Path) -> None:
    results = _hybrid_search_sync(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=False),
        top_k=2,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) <= 2


def test_hybrid_no_duplicates(db_path: Path, chroma_path: Path) -> None:
    results = _hybrid_search_sync(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids = [r.product_id for r in results]
    assert len(ids) == len(set(ids))


def test_hybrid_all_results_satisfy_filters(db_path: Path, chroma_path: Path) -> None:
    """Every returned HybridHit must satisfy the requested filters."""
    results = _hybrid_search_sync(
        "shirt dress shoes jacket trousers",
        SearchFilters(in_stock_only=True),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert all(r.product_id not in {"PROD-00002", "PROD-00005"} for r in results)


def test_hybrid_size_filter_enforced(db_path: Path, chroma_path: Path) -> None:
    """Size filter must exclude items with zero stock in that size.

    PROD-00001 has M=5 (passes); PROD-00003 has M=0 (must be excluded).
    """
    results = _hybrid_search_sync(
        "dress shirt",
        SearchFilters(size="M", in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids
    assert "PROD-00003" not in ids


def test_hybrid_item_in_both_methods_has_both_ranks(db_path: Path, chroma_path: Path) -> None:
    """A product found by both paths must have non-None keyword_rank and vector_rank."""
    results = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    # PROD-00001 contains "dress" in name → keyword hit
    # All products are vector hits (fake embeddings are identical)
    prod1 = next((r for r in results if r.product_id == "PROD-00001"), None)
    assert prod1 is not None
    assert prod1.keyword_rank is not None
    assert prod1.vector_rank is not None


def test_hybrid_results_are_deterministic(db_path: Path, chroma_path: Path) -> None:
    """Two identical calls must return the same product_id sequence."""
    r1 = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    r2 = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert [r.product_id for r in r1] == [r.product_id for r in r2]


def test_hybrid_item_in_both_ranks_above_item_in_one(db_path: Path, chroma_path: Path) -> None:
    """An item found by both methods must beat an item found by only one.

    PROD-00001 has "dress" in name → keyword hit.
    All items are vector hits (fake embeddings).
    PROD-00004 (Red Running Shoes) has no "dress" keyword → keyword miss.
    With equal weights PROD-00001 accumulates score from both lists.
    """
    results = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        n_candidates=10,
        rrf_k=60,
        kw_weight=1.0,
        vec_weight=1.0,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    ids = [r.product_id for r in results]
    # PROD-00001 should appear before any keyword-miss item
    keyword_misses = {r.product_id for r in results if r.keyword_rank is None}
    if keyword_misses:
        pos_prod1 = ids.index("PROD-00001")
        for miss_id in keyword_misses:
            assert pos_prod1 < ids.index(miss_id), (
                f"PROD-00001 (both lists) should rank above {miss_id} (vector only)"
            )


# ---------------------------------------------------------------------------
# Empty / stopword query → filter-only path
# ---------------------------------------------------------------------------


def test_hybrid_empty_query_returns_filter_only(db_path: Path, chroma_path: Path) -> None:
    """Empty query must bypass both methods and return filter-only results."""
    results = _hybrid_search_sync(
        "",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) == len(FIXTURE_PRODUCTS)
    # All have rrf_score=0.0 and no vector_rank
    assert all(r.rrf_score == 0.0 for r in results)
    assert all(r.vector_rank is None for r in results)


def test_hybrid_none_query_returns_filter_only(db_path: Path, chroma_path: Path) -> None:
    results = _hybrid_search_sync(
        None,
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    assert len(results) == len(FIXTURE_PRODUCTS)


def test_hybrid_stopwords_only_query_returns_filter_only(db_path: Path, chroma_path: Path) -> None:
    results = _hybrid_search_sync(
        "the and or a",
        SearchFilters(in_stock_only=True),
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=FAKE_EMBED_FN,
    )
    # in_stock_only=True: only PROD-00001, 00003, 00004 survive
    assert all(r.product_id in {"PROD-00001", "PROD-00003", "PROD-00004"} for r in results)


# ---------------------------------------------------------------------------
# One method returns nothing
# ---------------------------------------------------------------------------


def test_hybrid_vector_path_absent_still_returns_keyword_results(
    db_path: Path, tmp_path: Path
) -> None:
    """When the Chroma collection does not exist, keyword results are still returned."""
    empty_chroma = tmp_path / "empty_chroma"
    empty_chroma.mkdir()

    results = _hybrid_search_sync(
        "dress",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        chroma_path=empty_chroma,
        embed_fn=FAKE_EMBED_FN,
    )
    # Keyword path found something; vector path failed gracefully
    assert len(results) > 0
    assert all(r.vector_rank is None for r in results)
