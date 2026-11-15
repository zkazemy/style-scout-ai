"""Command-line entry points for StyleScout."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any


def build_catalog_cmd(args: argparse.Namespace) -> None:
    from stylescout.catalog.loader import build_catalog

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = build_catalog(sample_path=args.sample, db_path=args.db)
    print(
        f"Done — loaded: {summary['loaded']}, "
        f"dropped: {summary['dropped']}, "
        f"written: {summary['written']}"
    )


def build_index_cmd(args: argparse.Namespace) -> None:
    from stylescout.catalog.embeddings import build_index

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = build_index(db_path=args.db, chroma_path=args.chroma, collection_name=args.collection)
    print(f"Done — indexed: {summary['indexed']}")


# ---------------------------------------------------------------------------
# Search helpers (pure functions, testable without a real embedding model)
# ---------------------------------------------------------------------------

# Candidates fetched before filtering; headroom for tight filter conditions.
_N_CANDIDATES = 100


def parse_filters(args: argparse.Namespace) -> Any:
    """Build a SearchFilters from parsed CLI args."""
    from stylescout.catalog.search import SearchFilters

    return SearchFilters(
        category=args.category,
        color=args.color,
        occasion=args.occasion,
        brand=args.brand,
        min_price=args.min_price,
        max_price=args.max_price,
        size=args.size,
        in_stock_only=not args.include_out_of_stock,
    )


def run_keyword(
    query: str,
    filters: Any,
    top_k: int,
    db_path: str | Path = "data/catalog.db",
    n_candidates: int = _N_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return top_k keyword results that satisfy filters.

    Retrieves n_candidates rows via BM25 with no filter constraints, then
    re-checks all filters (including size) through _confirm_with_sqlite.
    This keeps filter logic in a single place shared with vector and hybrid modes.
    """
    from stylescout.catalog.hybrid import _confirm_with_sqlite
    from stylescout.catalog.search import SearchFilters, search_filtered

    # Fetch candidates with no filters so all items are reachable; the caller's
    # filters are enforced by _confirm_with_sqlite below (including in_stock_only).
    hits = search_filtered(
        query, SearchFilters(in_stock_only=False), db_path=db_path, limit=n_candidates
    )
    scores = {h.product_id: h.bm25_score for h in hits}
    ids = [h.product_id for h in hits]

    confirmed = _confirm_with_sqlite(ids, filters, Path(db_path))
    results = []
    for i, row in enumerate(confirmed[:top_k], 1):
        results.append(
            {
                "rank": i,
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "color": row["color"],
                "price_eur": row["price_eur"],
                "bm25_score": scores.get(row["product_id"], 0.0),
            }
        )
    return results


def run_vector(
    query: str,
    filters: Any,
    top_k: int,
    db_path: str | Path = "data/catalog.db",
    chroma_path: str | Path = "data/chroma",
    embed_fn: Any | None = None,
    n_candidates: int = _N_CANDIDATES,
) -> list[dict[str, Any]]:
    """Return top_k vector results that satisfy filters.

    Passes translatable filters as a Chroma where-clause to reduce candidate
    volume, then re-checks all filters (including size) through _confirm_with_sqlite.
    """
    from stylescout.catalog.embeddings import query_index
    from stylescout.catalog.hybrid import _build_chroma_where, _confirm_with_sqlite

    where = _build_chroma_where(filters)
    pairs = query_index(
        query=query,
        chroma_path=chroma_path,
        n_results=n_candidates,
        where=where,
        embed_fn=embed_fn,
    )
    distances = {pid: dist for pid, dist in pairs}
    ids = [pid for pid, _ in pairs]

    confirmed = _confirm_with_sqlite(ids, filters, Path(db_path))
    results = []
    for i, row in enumerate(confirmed[:top_k], 1):
        results.append(
            {
                "rank": i,
                "product_id": row["product_id"],
                "name": row["name"],
                "category": row["category"],
                "color": row["color"],
                "price_eur": row["price_eur"],
                "distance": distances.get(row["product_id"], 0.0),
            }
        )
    return results


def run_hybrid(
    query: str,
    filters: Any,
    top_k: int,
    db_path: str | Path = "data/catalog.db",
    chroma_path: str | Path = "data/chroma",
    embed_fn: Any | None = None,
) -> list[dict[str, Any]]:
    """Return top_k hybrid (RRF) results that satisfy filters."""
    from stylescout.catalog.hybrid import _hybrid_search_sync

    hits = _hybrid_search_sync(
        query=query,
        filters=filters,
        top_k=top_k,
        db_path=db_path,
        chroma_path=chroma_path,
        embed_fn=embed_fn,
    )
    return [
        {
            "rank": i,
            "product_id": h.product_id,
            "name": h.name,
            "category": h.category,
            "color": h.color,
            "price_eur": h.price_eur,
            "rrf_score": h.rrf_score,
            "keyword_rank": h.keyword_rank,
            "vector_rank": h.vector_rank,
        }
        for i, h in enumerate(hits, 1)
    ]


def format_table(results: list[dict[str, Any]], mode: str) -> str:
    """Render search results as a fixed-width text table.

    Pure function: no I/O, no DB access. Columns vary by mode so users
    can compare modes without noise from irrelevant fields.
    """
    if not results:
        return ""

    name_w = max(len(r["name"]) for r in results)
    name_w = max(name_w, 4)  # header minimum

    if mode == "hybrid":
        header = (
            f"{'#':>3}  {'Name':<{name_w}}  {'Category':<12}  {'Color':<10}"
            f"  {'Price':>8}  {'KW':>4}  {'Vec':>4}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in results:
            kw = str(r["keyword_rank"]) if r["keyword_rank"] is not None else "-"
            vec = str(r["vector_rank"]) if r["vector_rank"] is not None else "-"
            lines.append(
                f"{r['rank']:>3}  {r['name']:<{name_w}}  {r['category']:<12}"
                f"  {r['color']:<10}  {r['price_eur']:>7.2f}€  {kw:>4}  {vec:>4}"
            )
    elif mode == "vector":
        header = (
            f"{'#':>3}  {'Name':<{name_w}}  {'Category':<12}  {'Color':<10}"
            f"  {'Price':>8}  {'Dist':>6}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in results:
            lines.append(
                f"{r['rank']:>3}  {r['name']:<{name_w}}  {r['category']:<12}"
                f"  {r['color']:<10}  {r['price_eur']:>7.2f}€  {r['distance']:>6.3f}"
            )
    else:  # keyword
        header = (
            f"{'#':>3}  {'Name':<{name_w}}  {'Category':<12}  {'Color':<10}"
            f"  {'Price':>8}  {'BM25':>8}"
        )
        sep = "-" * len(header)
        lines = [header, sep]
        for r in results:
            lines.append(
                f"{r['rank']:>3}  {r['name']:<{name_w}}  {r['category']:<12}"
                f"  {r['color']:<10}  {r['price_eur']:>7.2f}€  {r['bm25_score']:>8.4f}"
            )

    return "\n".join(lines)


def search_cmd(args: argparse.Namespace, embed_fn: Any | None = None) -> None:
    """Run the search command. embed_fn is injectable for tests."""
    from chromadb.errors import NotFoundError

    from stylescout.catalog.search import CatalogIndexMissingError

    filters = parse_filters(args)
    db_path = args.db
    chroma_path = args.chroma
    mode = args.mode
    top_k = args.top_k
    query = args.query

    try:
        if mode == "keyword":
            results = run_keyword(query, filters, top_k, db_path=db_path)
        elif mode == "vector":
            results = run_vector(
                query, filters, top_k, db_path=db_path, chroma_path=chroma_path, embed_fn=embed_fn
            )
        else:
            results = run_hybrid(
                query, filters, top_k, db_path=db_path, chroma_path=chroma_path, embed_fn=embed_fn
            )
    except CatalogIndexMissingError:
        print("Catalog not found. Run: uv run stylescout build-catalog", file=sys.stderr)
        sys.exit(1)
    except NotFoundError:
        print("Vector index not found. Run: uv run stylescout build-index", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No products matched. Try relaxing the filters.")
        return

    table = format_table(results, mode)
    print(table)


# ---------------------------------------------------------------------------
# CLI wiring
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(prog="stylescout")
    sub = parser.add_subparsers(dest="command", required=True)

    bc = sub.add_parser("build-catalog", help="Load sample CSV and write catalog.db")
    bc.add_argument("--sample", default="data/sample_styles.csv")
    bc.add_argument("--db", default="data/catalog.db")
    bc.set_defaults(func=build_catalog_cmd)

    bi = sub.add_parser("build-index", help="Build Chroma vector index from catalog.db")
    bi.add_argument("--db", default="data/catalog.db")
    bi.add_argument("--chroma", default="data/chroma")
    bi.add_argument("--collection", default="products")
    bi.set_defaults(func=build_index_cmd)

    sr = sub.add_parser("search", help="Search the product catalog")
    sr.add_argument("query", help="Free-text search query")
    sr.add_argument(
        "--mode",
        choices=["hybrid", "keyword", "vector"],
        default="hybrid",
        help="Retrieval mode (default: hybrid)",
    )
    sr.add_argument(
        "--top-k", type=int, default=5, dest="top_k", help="Results to show (default: 5)"
    )
    sr.add_argument("--category", default=None)
    sr.add_argument("--color", default=None)
    sr.add_argument("--occasion", default=None)
    sr.add_argument("--brand", default=None)
    sr.add_argument("--min-price", type=float, default=None, dest="min_price")
    sr.add_argument("--max-price", type=float, default=None, dest="max_price")
    sr.add_argument("--size", default=None)
    sr.add_argument(
        "--include-out-of-stock",
        action="store_true",
        default=False,
        dest="include_out_of_stock",
        help="Include out-of-stock products (default: in-stock only)",
    )
    sr.add_argument("--db", default="data/catalog.db")
    sr.add_argument("--chroma", default="data/chroma")
    sr.set_defaults(func=search_cmd)

    args = parser.parse_args()
    args.func(args)
    sys.exit(0)


if __name__ == "__main__":
    main()
