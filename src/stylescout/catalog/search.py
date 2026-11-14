"""Filtered keyword search over the SQLite product catalog.

Strategy: FTS5 virtual table (products_fts) for bm25-ranked keyword matching
over name, brand, color, category, occasion, and description (spec §5.1);
structured SQL filters (category, occasion, price range, in_stock, size) are
applied as a JOIN predicate.

FTS index fields (searchable by keyword):
    name, brand, color, category, occasion, description

Filter-only fields (SQL WHERE predicates, not tokenised by FTS):
    price_eur, in_stock, sizes (via json_extract)
    category and occasion are also available as exact-match filters.

The FTS table is an external-content table backed by `products`, so it never
duplicates the actual column data. It must be rebuilt with
    INSERT INTO products_fts(products_fts) VALUES('rebuild')
whenever the products table changes. build_fts_index() does this and is called
by write_sqlite() in loader.py automatically.  build_fts_index() drops and
recreates the virtual table on each call so schema changes always take effect.

Filter-only fallback:
    When the query is empty, None, contains only stopwords, or consists
    entirely of tokens with no letter or digit, _to_fts_query() returns "".
    search_filtered() then skips the FTS MATCH clause and applies only the
    structured filters. A stopwords-only query with no active filters returns
    all products (up to `limit`), ordered by product_id.

FTS MATCH is safe against SQL injection because the query string is a bound
parameter. _require_fts_index() checks sqlite_master before the query so the
error message is clear and does not depend on SQLite's error text.
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

# SQL to create the FTS5 external-content table.
# Columns must match the names in `products` exactly (content= contract).
_CREATE_FTS_SQL = f"""
CREATE VIRTUAL TABLE IF NOT EXISTS {_FTS_TABLE}
USING fts5(
    product_id UNINDEXED,
    name,
    brand,
    color,
    category,
    occasion,
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
    """Raised when products_fts does not exist.

    Run `uv run stylescout build-catalog` to create it.
    """


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

    Drops the virtual table first so schema changes (added columns) always take
    effect. DROP TABLE IF EXISTS is a no-op when the table does not yet exist,
    so this is safe on a fresh database. After recreation, `rebuild` populates
    the index from the current `products` rows.
    """
    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {_FTS_TABLE}")
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
    color, category, occasion, and description (spec §5.1). Multi-word queries
    use OR semantics so "blue dress" finds products containing either word.
    Structured filters are applied as WHERE predicates on the joined products
    table so they never touch FTS internals.

    When query is None, empty, whitespace-only, contains only stopwords, or
    consists entirely of tokens with no letter or digit, the FTS clause is
    omitted and only the structured filters are applied (filter-only path).
    A stopwords-only query with no active filters returns all products
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

    db_path = Path(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if use_fts:
            _require_fts_index(conn, db_path)
        sql, params = _build_query(fts_query if use_fts else None, filters, limit)
        rows = conn.execute(sql, params).fetchall()

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
            bm25_score=row["bm25_score"] if use_fts else 0.0,
        )
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_fts_index(conn: sqlite3.Connection, db_path: Path) -> None:
    """Raise CatalogIndexMissingError when products_fts is absent.

    Checks sqlite_master directly rather than matching error message text.
    """
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (_FTS_TABLE,),
    ).fetchone()
    if row is None:
        raise CatalogIndexMissingError(
            f"FTS index missing at {db_path}; "
            "run `uv run stylescout build-catalog`."
        )


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
