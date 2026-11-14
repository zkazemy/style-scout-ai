"""Hybrid search: reciprocal rank fusion of keyword (BM25) and vector results.

Pipeline
--------
1. Keyword path  – search_filtered() → up to n_candidates ProductHits.
2. Vector path   – query_index()      → up to n_candidates (product_id, distance) pairs.
   Chroma metadata filters (category, color, occasion, price_eur, in_stock) are
   applied as an optimisation to keep Chroma results roughly on-target, but they
   are *not* relied upon for correctness.
3. RRF fusion    – rrf_fuse() merges the two ranked id lists into a single
   score-ranked list.
4. SQLite filter check – _confirm_with_sqlite() re-runs the full filter set
   (including size, which Chroma cannot evaluate) against the fused candidates
   and returns only those that pass, preserving RRF order.

The size filter exists only in SQLite (it is a JSON column) and cannot be
expressed as a Chroma where-clause.  All filter enforcement therefore happens
in step 4; the vector path may return non-size-filtered candidates, and they
are discarded then.  n_candidates is intentionally larger than top_k (default
50 vs. 10) to give headroom when filters are tight.

Async surface
-------------
The public async def hybrid_search() wraps a synchronous core
(_hybrid_search_sync) via asyncio.to_thread so the event loop is not blocked.
The sync core is callable directly from tests without an event loop.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from stylescout.catalog.embeddings import EmbeddingFn, query_index
from stylescout.catalog.search import SearchFilters, _to_fts_query, search_filtered
from stylescout.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public models
# ---------------------------------------------------------------------------


class HybridHit(BaseModel):
    """One result from hybrid_search.

    keyword_rank and vector_rank are 1-based positions in their respective
    candidate lists (None when the item did not appear in that list).
    """

    product_id: str
    name: str
    category: str
    color: str
    price_eur: float
    rrf_score: float
    keyword_rank: int | None
    vector_rank: int | None


# ---------------------------------------------------------------------------
# Pure RRF function
# ---------------------------------------------------------------------------


def rrf_fuse(
    rankings: list[list[str]],
    weights: list[float],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion of multiple ranked id lists.

    score(id) = sum over lists of  weight_i / (k + rank_i)
    where rank starts at 1 and items absent from a list contribute 0.

    Ties are broken by product_id (ascending) for determinism.

    Returns a list of (product_id, score) sorted descending by score.

    Args:
        rankings: Each inner list is a ranked sequence of product ids, best first.
        weights:  Per-list weight, parallel to rankings.  Must be same length.
        k:        Rank-smoothing constant (standard default: 60).
    """
    if len(rankings) != len(weights):
        raise ValueError("rankings and weights must have the same length")

    scores: dict[str, float] = {}
    for ranked_list, weight in zip(rankings, weights):
        for rank_0, pid in enumerate(ranked_list):
            scores[pid] = scores.get(pid, 0.0) + weight / (k + rank_0 + 1)

    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_chroma_where(filters: SearchFilters) -> dict[str, Any] | None:
    """Translate SearchFilters into a Chroma metadata where-clause.

    Only the fields stored in Chroma's METADATA_FIELDS are included:
    category, color, occasion, price_eur, in_stock.

    Size is absent from Chroma metadata and is therefore not translated here;
    it is enforced in the SQLite filter check after fusion.

    Returns None when no translatable filters are active (avoids passing an
    empty where dict to Chroma, which raises an error).
    """
    clauses: list[dict[str, Any]] = []

    if filters.category is not None:
        clauses.append({"category": {"$eq": filters.category.lower()}})
    if filters.color is not None:
        clauses.append({"color": {"$eq": filters.color.lower()}})
    if filters.occasion is not None:
        clauses.append({"occasion": {"$eq": filters.occasion.lower()}})
    if filters.min_price is not None:
        clauses.append({"price_eur": {"$gte": filters.min_price}})
    if filters.max_price is not None:
        clauses.append({"price_eur": {"$lte": filters.max_price}})
    if filters.in_stock_only:
        clauses.append({"in_stock": {"$eq": 1}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _confirm_with_sqlite(
    candidate_ids: list[str],
    filters: SearchFilters,
    db_path: Path,
) -> list[dict[str, Any]]:
    """Return the subset of candidate_ids that pass all filters, in RRF order.

    Runs a single parameterized IN query.  Result rows are re-ordered to match
    the input candidate_ids order (RRF score order) rather than SQLite's
    arbitrary row order.

    This is the authoritative filter check: every constraint (including size,
    which Chroma cannot evaluate) is enforced here.
    """
    if not candidate_ids:
        return []

    placeholders = ",".join("?" * len(candidate_ids))
    where_clauses = [f"product_id IN ({placeholders})"]
    params: list[Any] = list(candidate_ids)

    if filters.category is not None:
        where_clauses.append("category = ?")
        params.append(filters.category.lower())
    if filters.color is not None:
        where_clauses.append("color = ?")
        params.append(filters.color.lower())
    if filters.occasion is not None:
        where_clauses.append("occasion = ?")
        params.append(filters.occasion.lower())
    if filters.brand is not None:
        where_clauses.append("brand = ?")
        params.append(filters.brand)
    if filters.min_price is not None:
        where_clauses.append("price_eur >= ?")
        params.append(filters.min_price)
    if filters.max_price is not None:
        where_clauses.append("price_eur <= ?")
        params.append(filters.max_price)
    if filters.in_stock_only:
        where_clauses.append("in_stock = 1")
    if filters.size is not None:
        where_clauses.append("CAST(json_extract(sizes, ?) AS INTEGER) > 0")
        params.append(f"$.{filters.size}")

    sql = "SELECT product_id, name, category, color, price_eur FROM products WHERE " + " AND ".join(
        where_clauses
    )

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()

    by_id = {row["product_id"]: dict(row) for row in rows}
    # Preserve RRF order; skip ids that did not survive the filter check
    return [by_id[pid] for pid in candidate_ids if pid in by_id]


# ---------------------------------------------------------------------------
# Synchronous core
# ---------------------------------------------------------------------------


def _hybrid_search_sync(
    query: str | None,
    filters: SearchFilters,
    top_k: int = 10,
    n_candidates: int | None = None,
    rrf_k: int | None = None,
    kw_weight: float | None = None,
    vec_weight: float | None = None,
    db_path: str | Path = "data/catalog.db",
    chroma_path: str | Path = "data/chroma",
    embed_fn: EmbeddingFn | None = None,
) -> list[HybridHit]:
    """Synchronous hybrid search core; callable directly from tests.

    When query is None, empty, or reduces to no FTS tokens after stopword
    removal, both retrieval methods are skipped and the filter-only results
    from search_filtered are returned as HybridHit rows (keyword_rank set,
    vector_rank None, rrf_score 0.0).

    When only one method returns results, RRF uses only that list; the other
    method's rank field is None for every result.

    n_candidates, rrf_k, kw_weight, vec_weight default to the values in
    config.py so callers only override what they need.
    """
    n_candidates = n_candidates if n_candidates is not None else settings.hybrid_n_candidates
    rrf_k = rrf_k if rrf_k is not None else settings.rrf_k
    kw_weight = kw_weight if kw_weight is not None else settings.rrf_kw_weight
    vec_weight = vec_weight if vec_weight is not None else settings.rrf_vec_weight

    db_path = Path(db_path)
    chroma_path = Path(chroma_path)

    # Fall back to filter-only when there is no useful keyword query
    fts_query = _to_fts_query(query or "")
    if not fts_query:
        filter_only = search_filtered(query, filters, db_path=db_path, limit=top_k)
        return [
            HybridHit(
                product_id=h.product_id,
                name=h.name,
                category=h.category,
                color=h.color,
                price_eur=h.price_eur,
                rrf_score=0.0,
                keyword_rank=i + 1,
                vector_rank=None,
            )
            for i, h in enumerate(filter_only)
        ]

    # --- Keyword path ---
    kw_hits = search_filtered(query, filters, db_path=db_path, limit=n_candidates)
    kw_ids: list[str] = [h.product_id for h in kw_hits]

    # --- Vector path ---
    where = _build_chroma_where(filters)
    try:
        vec_pairs = query_index(
            query=query,
            chroma_path=chroma_path,
            n_results=n_candidates,
            where=where,
            embed_fn=embed_fn,
        )
        vec_ids: list[str] = [pid for pid, _ in vec_pairs]
    except Exception:
        logger.warning("hybrid_search: vector path failed; using keyword-only", exc_info=True)
        vec_ids = []

    # --- RRF fusion ---
    fused = rrf_fuse(
        rankings=[kw_ids, vec_ids],
        weights=[kw_weight, vec_weight],
        k=rrf_k,
    )
    fused_ids = [pid for pid, _ in fused]
    fused_scores = {pid: score for pid, score in fused}

    # Build lookup tables for rank fields
    kw_rank_of = {pid: i + 1 for i, pid in enumerate(kw_ids)}
    vec_rank_of = {pid: i + 1 for i, pid in enumerate(vec_ids)}

    # --- SQLite filter check (authoritative; enforces size) ---
    confirmed = _confirm_with_sqlite(fused_ids, filters, db_path)

    results: list[HybridHit] = []
    for row in confirmed:
        pid = row["product_id"]
        results.append(
            HybridHit(
                product_id=pid,
                name=row["name"],
                category=row["category"],
                color=row["color"],
                price_eur=row["price_eur"],
                rrf_score=fused_scores[pid],
                keyword_rank=kw_rank_of.get(pid),
                vector_rank=vec_rank_of.get(pid),
            )
        )
        if len(results) == top_k:
            break

    return results


# ---------------------------------------------------------------------------
# Public async wrapper (spec §5.2)
# ---------------------------------------------------------------------------


async def hybrid_search(
    query: str | None,
    filters: SearchFilters,
    top_k: int = 10,
    n_candidates: int | None = None,
    rrf_k: int | None = None,
    kw_weight: float | None = None,
    vec_weight: float | None = None,
    db_path: str | Path = "data/catalog.db",
    chroma_path: str | Path = "data/chroma",
    embed_fn: EmbeddingFn | None = None,
) -> list[HybridHit]:
    """Async wrapper around _hybrid_search_sync.

    Runs the synchronous core in a thread pool via asyncio.to_thread so the
    FastAPI event loop is not blocked during disk and Chroma I/O.
    """
    return await asyncio.to_thread(
        _hybrid_search_sync,
        query,
        filters,
        top_k,
        n_candidates,
        rrf_k,
        kw_weight,
        vec_weight,
        db_path,
        chroma_path,
        embed_fn,
    )
