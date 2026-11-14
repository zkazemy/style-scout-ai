"""Filtered keyword search over the SQLite product catalog.

Strategy: FTS5 virtual table (products_fts) for bm25-ranked keyword matching
over name, brand, color, and description; structured SQL filters (category,
occasion, price range, in_stock, size) are applied as a JOIN predicate.

FTS index fields (searchable by keyword):
    name, brand, color, description

Filter-only fields (SQL WHERE predicates, not tokenised by FTS):
    category, occasion, price_eur, in_stock, sizes (via json_extract)

The FTS table is an external-content table backed by `products`, so it never
duplicates the actual column data. It must be rebuilt with
    INSERT INTO products_fts(products_fts) VALUES('rebuild')
whenever the products table changes. build_fts_index() does this and is called
by write_sqlite() in loader.py automatically.

Filter-only fallback:
    When the query is empty, None, contains only stopwords, or consists
    entirely of tokens with no letter or digit, _to_fts_query() returns "".
    search_filtered() then skips the FTS MATCH clause and applies only the
    structured filters. A stopwords-only query with no active filters returns
    all products (up to `limit`), ordered by product_id.

FTS MATCH is safe against SQL injection because the query string is a bound
parameter. Malformed FTS queries raise sqlite3.OperationalError; the
OperationalError handler falls back to filter-only rather than crashing.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Common English stopwords and SQL DDL keywords (never useful for fashion search).
_STOPWORDS: frozenset[str] = frozenset(
    "a an the and or not in on at to for of with is are was be"
    " it its this that these those he she they we you i"
    " drop table select insert update delete create alter from where".split()
)

# FTS virtual table name
_FTS_TABLE = "products_fts"

# SQL to create the FTS5 external-content table
_CREATE_FTS_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
USING fts5(
    product_id UNINDEXED,
    name,
    brand,
    color,
    description,
    content=products,
    content_rowid=rowid
)
"""

_REBUILD_FTS_SQL = f"INSERT INTO {_FTS_TABLE}({_FTS_TABLE}) VALUES('rebuild')"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class CatalogIndexMissingError(RuntimeError):
    """Raised when products_fts does not exist; call build_fts_index() first."""


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class SearchFilters(BaseModel):
    """Structured filters applied as SQL WHERE predicates."""

    category: str | None = None
    color: str | None = None
    min_price: float | None = None
    max_price: float | None = None
    occasion: str | None = None
    brand: str | None = None
    size: str | None = None
    in_stock_only: bool = True


class ProductHit(BaseModel):
    """One result row returned by search_filtered."""

    product_id: str
    name: str
    category: str
    color: str
    brand: str
    occasion: str
    price_eur: float
    in_stock: int
    bm25_score: float


# ---------------------------------------------------------------------------
# FTS index management
# ---------------------------------------------------------------------------


def build_fts_index(db_path: str | Path) -> None:
    """Create (or rebuild) the FTS5 index for the products table.

    Safe to call repeatedly: CREATE VIRTUAL TABLE IF NOT EXISTS is a no-op when
    the table exists, and `rebuild` re-indexes all current rows.
    """
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_CREATE_FTS_SQL)
        conn.execute(_REBUILD_FTS_SQL)
        conn.commit()
    logger.info("build_fts_index: FTS5 index rebuilt at %s", db_path)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def search_filtered(
    query: str | None,
    filters: SearchFilters,
    db_path: str | Path = "data/catalog.db",
    limit: int = 20,
) -> list[ProductHit]:
    """Keyword search ranked by bm25, with structured filters.

    The FTS MATCH clause ranks results by term frequency across name, brand,
    color, and description. Multi-word queries use OR semantics (any term
    matches) so "blue dress" finds products containing "blue" or "dress".
    Structured filters are applied as WHERE predicates on the joined products
    table so they never touch FTS internals.

    When query is None, empty, whitespace-only, contains only stopwords, or
    consists entirely of tokens with no letter or digit, the FTS clause is
    omitted entirely and only the structured filters are applied (filter-only
    path). A stopwords-only query with no active filters returns all products
    (up to `limit`), ordered by product_id.

    Returns up to `limit` ProductHit rows:
    - FTS path: ordered best-match first (most negative bm25 score).
    - Filter-only path: ordered by product_id ascending.

    Size availability is checked with json_extract: sizes is a JSON object
    mapping size label → stock count (e.g. {"M": 4, "L": 0}).

    Raises CatalogIndexMissingError if products_fts does not exist.
    """
    fts_query = _to_fts_query(query or "")
    use_fts = bool(fts_query)

    sql, params = _build_query(fts_query if use_fts else None, filters, limit)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc) and "products_fts" in str(exc):
            raise CatalogIndexMissingError(
                f"FTS index missing at {db_path}; call build_fts_index() first."
            ) from exc
        raise

    bm25_placeholder = 0.0
    return [
        ProductHit(
            product_id=row["product_id"],
            name=row["name"],
            category=row["category"],
            color=row["color"],
            brand=row["brand"],
            occasion=row["occasion"],
            price_eur=row["price_eur"],
            in_stock=row["in_stock"],
            bm25_score=row["bm25_score"] if use_fts else bm25_placeholder,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_fts_query(query: str | None) -> str:
    """Convert a user query string to an FTS5 MATCH expression.

    Processing steps (in order):
    1. Split on whitespace.
    2. Strip characters that are neither a letter nor a digit from each token.
    3. Drop tokens that, after stripping, contain no letter or digit.
    4. Drop stopwords and SQL DDL keywords (case-insensitive).
    5. Join survivors with OR for broad matching semantics.

    Returns "" when no valid tokens remain so the caller uses filter-only mode.
    """
    if not query:
        return ""
    tokens = []
    for raw in query.split():
        # Keep only alphanumeric chars and common word chars; remove everything else
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", raw)
        if not cleaned:
            continue
        if cleaned.lower() in _STOPWORDS:
            continue
        tokens.append(cleaned)
    if not tokens:
        return ""
    return " OR ".join(tokens)


def _build_query(
    fts_query: str | None, filters: SearchFilters, limit: int
) -> tuple[str, list[Any]]:
    """Return (sql, params) for either FTS+filter search or filter-only search.

    When fts_query is None, the FTS table is not touched: the query reads
    directly from `products` with only the structured WHERE predicates.
    When fts_query is a non-empty string, products_fts is joined for bm25
    ranking.
    """
    params: list[Any] = []
    where_clauses: list[str] = []

    if fts_query is not None:
        where_clauses.append("products_fts MATCH ?")
        params.append(fts_query)

    if filters.category is not None:
        where_clauses.append("p.category = ?")
        params.append(filters.category.lower())

    if filters.color is not None:
        where_clauses.append("p.color = ?")
        params.append(filters.color.lower())

    if filters.occasion is not None:
        where_clauses.append("p.occasion = ?")
        params.append(filters.occasion.lower())

    if filters.brand is not None:
        where_clauses.append("p.brand = ?")
        params.append(filters.brand)

    if filters.min_price is not None:
        where_clauses.append("p.price_eur >= ?")
        params.append(filters.min_price)

    if filters.max_price is not None:
        where_clauses.append("p.price_eur <= ?")
        params.append(filters.max_price)

    if filters.in_stock_only:
        where_clauses.append("p.in_stock = 1")

    if filters.size is not None:
        # json_extract returns the stock count for the requested size label;
        # cast to INTEGER to treat NULL (key absent) as 0.
        where_clauses.append("CAST(json_extract(p.sizes, ?) AS INTEGER) > 0")
        params.append(f"$.{filters.size}")

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    if fts_query is not None:
        sql = f"""
            SELECT
                p.product_id,
                p.name,
                p.category,
                p.color,
                p.brand,
                p.occasion,
                p.price_eur,
                p.in_stock,
                bm25(products_fts) AS bm25_score
            FROM products_fts
            JOIN products AS p ON products_fts.rowid = p.rowid
            {where_sql}
            ORDER BY bm25_score
            LIMIT ?
        """
    else:
        sql = f"""
            SELECT
                p.product_id,
                p.name,
                p.category,
                p.color,
                p.brand,
                p.occasion,
                p.price_eur,
                p.in_stock,
                0.0 AS bm25_score
            FROM products AS p
            {where_sql}
            ORDER BY p.product_id
            LIMIT ?
        """

    params.append(limit)
    return sql, params
