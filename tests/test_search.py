"""Tests for catalog/search.py.

All tests use an in-memory SQLite database populated with a small fixture set
so they run offline with no file I/O beyond the temp directory.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from stylescout.catalog.search import (
    CatalogIndexMissingError,
    ProductHit,
    SearchFilters,
    _to_fts_query,
    build_fts_index,
    search_filtered,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURE_PRODUCTS: list[dict[str, Any]] = [
    {
        "product_id": "PROD-00001",
        "name": "Blue Floral Dress",
        "category": "dress",
        "color": "blue",
        "brand": "Zara",
        "occasion": "casual",
        "description": "A light summer dress with floral print.",
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
        "description": "Classic black leather biker jacket.",
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
        "description": "Crisp white button-down shirt for office wear.",
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
        "description": "Lightweight running shoes with cushioned sole.",
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
        "description": "Slim-fit navy chinos for smart casual outfits.",
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
    (product_id, name, category, color, brand, occasion, description, price_eur, sizes, in_stock)
VALUES
    (:product_id, :name, :category, :color, :brand, :occasion, :description,
     :price_eur, :sizes, :in_stock)
"""


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    """Write fixture products to a temp SQLite DB and build the FTS index."""
    path = tmp_path / "test_catalog.db"
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE_TABLE_SQL)
        conn.executemany(_INSERT_SQL, FIXTURE_PRODUCTS)
        conn.commit()
    build_fts_index(path)
    return path


# ---------------------------------------------------------------------------
# Tests: build_fts_index
# ---------------------------------------------------------------------------


def test_build_fts_index_creates_table(db_path: Path) -> None:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='products_fts'"
        ).fetchall()
    assert len(rows) == 1


def test_build_fts_index_idempotent(db_path: Path) -> None:
    """Calling build_fts_index twice must not raise and table count stays consistent."""
    build_fts_index(db_path)  # second call
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM products_fts").fetchone()[0]
    assert count == len(FIXTURE_PRODUCTS)


# ---------------------------------------------------------------------------
# Tests: search_filtered — basic keyword matching
# ---------------------------------------------------------------------------


def test_search_returns_product_hit_instances(db_path: Path) -> None:
    results = search_filtered("dress", SearchFilters(in_stock_only=False), db_path=db_path)
    assert len(results) > 0
    assert all(isinstance(r, ProductHit) for r in results)


def test_search_dress_keyword(db_path: Path) -> None:
    results = search_filtered("dress", SearchFilters(in_stock_only=False), db_path=db_path)
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids


def test_search_keyword_in_description(db_path: Path) -> None:
    results = search_filtered("floral", SearchFilters(in_stock_only=False), db_path=db_path)
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids


def test_search_by_brand_name(db_path: Path) -> None:
    results = search_filtered("Nike", SearchFilters(in_stock_only=False), db_path=db_path)
    ids = {r.product_id for r in results}
    assert "PROD-00004" in ids


def test_search_results_ordered_by_bm25(db_path: Path) -> None:
    """bm25 scores are negative; ORDER BY bm25_score puts best match first (most negative)."""
    results = search_filtered("dress", SearchFilters(in_stock_only=False), db_path=db_path)
    scores = [r.bm25_score for r in results]
    assert scores == sorted(scores)  # ascending = best (most negative) first


# ---------------------------------------------------------------------------
# Tests: search_filtered — structured filters
# ---------------------------------------------------------------------------


def test_filter_category(db_path: Path) -> None:
    results = search_filtered(
        "shirt", SearchFilters(category="shirt", in_stock_only=False), db_path=db_path
    )
    assert all(r.category == "shirt" for r in results)
    assert any(r.product_id == "PROD-00003" for r in results)


def test_filter_in_stock_only_excludes_out_of_stock(db_path: Path) -> None:
    # Search across all categories; PROD-00002 (jacket) and PROD-00005 (trousers) are out of stock
    results = search_filtered(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=True),
        db_path=db_path,
    )
    assert len(results) > 0
    assert all(r.in_stock == 1 for r in results)
    out_of_stock_ids = {"PROD-00002", "PROD-00005"}
    assert not out_of_stock_ids.intersection({r.product_id for r in results})


def test_filter_in_stock_false_includes_out_of_stock(db_path: Path) -> None:
    results = search_filtered("jacket", SearchFilters(in_stock_only=False), db_path=db_path)
    assert any(r.product_id == "PROD-00002" for r in results)


def test_filter_max_price(db_path: Path) -> None:
    results = search_filtered(
        "dress shirt trousers shoes jacket",
        SearchFilters(max_price=50.0, in_stock_only=False),
        db_path=db_path,
    )
    assert all(r.price_eur <= 50.0 for r in results)


def test_filter_min_price(db_path: Path) -> None:
    results = search_filtered(
        "dress shirt trousers shoes jacket",
        SearchFilters(min_price=80.0, in_stock_only=False),
        db_path=db_path,
    )
    assert all(r.price_eur >= 80.0 for r in results)


def test_filter_price_range(db_path: Path) -> None:
    results = search_filtered(
        "dress shirt trousers shoes jacket",
        SearchFilters(min_price=40.0, max_price=100.0, in_stock_only=False),
        db_path=db_path,
    )
    assert all(40.0 <= r.price_eur <= 100.0 for r in results)


def test_filter_color(db_path: Path) -> None:
    results = search_filtered(
        "blue dress", SearchFilters(color="blue", in_stock_only=False), db_path=db_path
    )
    assert all(r.color == "blue" for r in results)


def test_filter_occasion(db_path: Path) -> None:
    results = search_filtered(
        "formal shirt", SearchFilters(occasion="formal", in_stock_only=False), db_path=db_path
    )
    assert all(r.occasion == "formal" for r in results)


def test_filter_size_with_stock(db_path: Path) -> None:
    """Size filter: only rows where sizes JSON has a positive count for that size."""
    results = search_filtered(
        "dress shirt", SearchFilters(size="M", in_stock_only=False), db_path=db_path
    )
    # PROD-00001 has M: 5 (in stock); PROD-00003 has M: 0 (out of stock for M)
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids
    assert "PROD-00003" not in ids


def test_filter_brand(db_path: Path) -> None:
    results = search_filtered(
        "shoes", SearchFilters(brand="Nike", in_stock_only=False), db_path=db_path
    )
    assert all(r.brand == "Nike" for r in results)
    assert any(r.product_id == "PROD-00004" for r in results)


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


def test_empty_query_falls_back_to_filter_only(db_path: Path) -> None:
    """Empty query triggers filter-only path; all products returned when no other filters."""
    results = search_filtered("", SearchFilters(in_stock_only=False), db_path=db_path)
    assert len(results) == len(FIXTURE_PRODUCTS)


def test_blank_query_falls_back_to_filter_only(db_path: Path) -> None:
    """Blank query triggers filter-only path; all products returned when no other filters."""
    results = search_filtered("   ", SearchFilters(in_stock_only=False), db_path=db_path)
    assert len(results) == len(FIXTURE_PRODUCTS)


def test_only_punctuation_query_falls_back_to_filter_only(db_path: Path) -> None:
    """A query of only single quotes produces no tokens; must fall back to filter-only."""
    results = search_filtered("'''", SearchFilters(in_stock_only=False), db_path=db_path)
    assert len(results) == len(FIXTURE_PRODUCTS)


def test_no_matches_returns_empty(db_path: Path) -> None:
    results = search_filtered("helicopter", SearchFilters(in_stock_only=False), db_path=db_path)
    assert results == []


def test_limit_respected(db_path: Path) -> None:
    results = search_filtered(
        "dress shirt jacket shoes trousers",
        SearchFilters(in_stock_only=False),
        db_path=db_path,
        limit=2,
    )
    assert len(results) <= 2


# ---------------------------------------------------------------------------
# Tests: _to_fts_query
# ---------------------------------------------------------------------------


def test_to_fts_query_single_word() -> None:
    result = _to_fts_query("dress")
    assert result != ""
    assert "dress" in result


def test_to_fts_query_multiple_words_joined_with_or() -> None:
    result = _to_fts_query("blue dress")
    assert "blue" in result
    assert "dress" in result
    assert " OR " in result


def test_to_fts_query_strips_single_quotes() -> None:
    result = _to_fts_query("'sneaker'")
    assert "'" not in result
    assert "sneaker" in result


def test_to_fts_query_empty_returns_empty() -> None:
    assert _to_fts_query("") == ""
    assert _to_fts_query("   ") == ""
    assert _to_fts_query(None) == ""  # type: ignore[arg-type]


def test_to_fts_query_only_quotes_returns_empty() -> None:
    assert _to_fts_query("'''") == ""


def test_to_fts_query_no_letter_or_digit_tokens_dropped() -> None:
    """Tokens with no alphanumeric character are dropped entirely."""
    assert _to_fts_query("*") == ""
    assert _to_fts_query("( )") == ""
    assert _to_fts_query('""') == ""
    # A token that contains alnum is kept even if it has special chars mixed in
    result = _to_fts_query("dress*")
    assert "dress" in result


def test_to_fts_query_stopwords_dropped() -> None:
    """Pure stopwords produce an empty result."""
    assert _to_fts_query("the and or a") == ""
    # Non-stopword alongside stopword: only the content word survives
    result = _to_fts_query("the dress")
    assert "dress" in result
    assert "the" not in result.lower().replace('"', "")


# ---------------------------------------------------------------------------
# Bug 1: None / empty / stopwords-only → filter-only path
# ---------------------------------------------------------------------------


def test_none_query_returns_filter_only_results(db_path: Path) -> None:
    """query=None must not crash; must return all rows matching the filters."""
    results = search_filtered(  # type: ignore[arg-type]
        None, SearchFilters(category="dress", in_stock_only=False), db_path=db_path
    )
    assert len(results) > 0
    assert all(r.category == "dress" for r in results)


def test_none_query_no_filters_returns_all_products(db_path: Path) -> None:
    """query=None with no active filters returns all products in deterministic order."""
    results = search_filtered(None, SearchFilters(in_stock_only=False), db_path=db_path)  # type: ignore[arg-type]
    assert len(results) == len(FIXTURE_PRODUCTS)


def test_filter_only_order_is_deterministic(db_path: Path) -> None:
    """Two consecutive filter-only calls return the same product_id sequence."""
    r1 = search_filtered(None, SearchFilters(in_stock_only=False), db_path=db_path)  # type: ignore[arg-type]
    r2 = search_filtered(None, SearchFilters(in_stock_only=False), db_path=db_path)  # type: ignore[arg-type]
    assert [r.product_id for r in r1] == [r.product_id for r in r2]


def test_empty_query_with_filters_returns_filter_only_results(db_path: Path) -> None:
    """Empty string falls back to filter-only: returns rows matching the filters."""
    results = search_filtered(
        "", SearchFilters(category="shirt", in_stock_only=False), db_path=db_path
    )
    assert len(results) > 0
    assert all(r.category == "shirt" for r in results)


def test_whitespace_only_query_with_filters_returns_filter_only_results(db_path: Path) -> None:
    """Whitespace-only query falls back to filter-only results."""
    results = search_filtered("   ", SearchFilters(in_stock_only=True), db_path=db_path)
    assert len(results) > 0
    assert all(r.in_stock == 1 for r in results)


def test_stopwords_only_query_returns_filter_only_results(db_path: Path) -> None:
    """A query of only stopwords falls back to filter-only.

    A stopwords-only query with no filters returns all products (up to limit).
    """
    results = search_filtered(
        "the and or a", SearchFilters(category="shoes", in_stock_only=False), db_path=db_path
    )
    assert len(results) > 0
    assert all(r.category == "shoes" for r in results)


def test_stopwords_only_no_filters_returns_all_products(db_path: Path) -> None:
    """Stopwords-only query with no active filters returns all products (filter-only, no WHERE)."""
    results = search_filtered("the and or a", SearchFilters(in_stock_only=False), db_path=db_path)
    assert len(results) == len(FIXTURE_PRODUCTS)


# ---------------------------------------------------------------------------
# Bug 2a: missing FTS table → CatalogIndexMissingError
# ---------------------------------------------------------------------------


def test_missing_fts_table_raises_catalog_index_missing_error(tmp_path: Path) -> None:
    """If products_fts does not exist, search_filtered must raise CatalogIndexMissingError."""
    db = tmp_path / "no_index.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE products (
                product_id TEXT PRIMARY KEY, name TEXT, category TEXT, color TEXT,
                brand TEXT, occasion TEXT, description TEXT, price_eur REAL,
                sizes TEXT, in_stock INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO products VALUES ('P1','T','dress','blue','X','casual','D',50.0,'{}',1)"
        )
    with pytest.raises(CatalogIndexMissingError):
        search_filtered("dress", SearchFilters(in_stock_only=False), db_path=db)


# ---------------------------------------------------------------------------
# Bug 2b: FTS special characters in query → safe, still returns real matches
# ---------------------------------------------------------------------------


def test_fts_special_chars_query_returns_safely(db_path: Path) -> None:
    """FTS special characters must never raise, and real words alongside them still match.

    Each case pairs a dangerous token with a real keyword present in the fixture.
    The assertion checks that at least one result is returned, proving the real
    word survived sanitization and the special character did not abort the query.
    """
    cases: list[tuple[str, str]] = [
        ('"sneakers" dress', "PROD-00001"),       # double-quoted phrase + real word
        ("black* jacket", "PROD-00002"),           # wildcard prefix + real word (in_stock=0)
        ("(dress OR shirt) dress", "PROD-00001"),  # parenthesised boolean + real word
        ("category:dress dress", "PROD-00001"),    # column specifier + real word
        ("AND OR NOT dress", "PROD-00001"),         # bare FTS keywords + real word
        ("dress AND", "PROD-00001"),                # trailing operator
        ("NEAR dress", "PROD-00001"),               # NEAR keyword
        ("NEAR( dress", "PROD-00001"),              # NEAR with open paren
    ]
    for query, expected_id in cases:
        results = search_filtered(query, SearchFilters(in_stock_only=False), db_path=db_path)
        ids = {r.product_id for r in results}
        assert expected_id in ids, (
            f"Expected {expected_id!r} in results for query {query!r}, got {ids}"
        )


def test_all_punctuation_query_returns_filter_only(db_path: Path) -> None:
    """Queries with no alphanumeric tokens fall back to filter-only, not empty."""
    for q in ['"', '""', "* ( )"]:
        results = search_filtered(q, SearchFilters(in_stock_only=False), db_path=db_path)
        assert len(results) == len(FIXTURE_PRODUCTS), (
            f"Expected all {len(FIXTURE_PRODUCTS)} products for query {q!r}, "
            f"got {len(results)}"
        )


def test_injection_attempt_returns_normally(db_path: Path) -> None:
    """SQL-injection attempt must not raise, must return a list, and must leave
    the products table intact.  The query contains real words (DROP, TABLE,
    products) so it takes the keyword path; what matters is no exception and no
    data loss.
    """
    injection = "'; DROP TABLE products; --"
    result = search_filtered(injection, SearchFilters(in_stock_only=False), db_path=db_path)
    assert isinstance(result, list)
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert count == len(FIXTURE_PRODUCTS)


def test_mixed_punctuation_and_word_returns_word_matches(db_path: Path) -> None:
    """'dress \"' — the real word survives stripping and returns dress results."""
    results = search_filtered('dress "', SearchFilters(in_stock_only=False), db_path=db_path)
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids


# ---------------------------------------------------------------------------
# Bug 2c: size filter excludes zero-stock sizes
# ---------------------------------------------------------------------------


def test_size_filter_excludes_zero_stock_for_that_size(db_path: Path) -> None:
    """Size M: PROD-00001 has M=5 (included); PROD-00003 has M=0 (excluded)."""
    results = search_filtered(
        "dress shirt", SearchFilters(size="M", in_stock_only=False), db_path=db_path
    )
    ids = {r.product_id for r in results}
    assert "PROD-00001" in ids
    assert "PROD-00003" not in ids
