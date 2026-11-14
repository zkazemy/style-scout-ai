"""Load, clean, enrich, and persist the fashion product catalog.

Pipeline:
    load_sample -> clean -> enrich -> write_sqlite
All steps are pure functions; side effects are confined to write_sqlite.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants shared with generator.py
# ---------------------------------------------------------------------------

PRICE_RANGE: dict[str, tuple[float, float]] = {
    "dress": (30.0, 200.0),
    "shirt": (20.0, 120.0),
    "trousers": (25.0, 150.0),
    "shoes": (40.0, 250.0),
    "jacket": (50.0, 350.0),
    "accessory": (10.0, 80.0),
}

SIZES_BY_CATEGORY: dict[str, list[str]] = {
    "dress": ["XS", "S", "M", "L", "XL"],
    "shirt": ["XS", "S", "M", "L", "XL", "XXL"],
    "trousers": ["28", "30", "32", "34", "36", "38"],
    "shoes": ["36", "37", "38", "39", "40", "41", "42", "43"],
    "jacket": ["XS", "S", "M", "L", "XL"],
    "accessory": ["ONE"],
}

# Map Kaggle usage values to our occasion vocabulary
USAGE_TO_OCCASION: dict[str, str] = {
    "casual": "casual",
    "formal": "formal",
    "sports": "sport",
    "ethnic": "casual",
    "party": "evening",
    "smart casual": "casual",
    "travel": "casual",
    "home": "casual",
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS products (
    product_id          TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    category            TEXT NOT NULL,
    color               TEXT NOT NULL,
    brand               TEXT NOT NULL,
    occasion            TEXT,
    description         TEXT,
    price_eur           REAL NOT NULL,
    sizes               TEXT NOT NULL,
    in_stock            INTEGER NOT NULL
)
"""

INSERT_SQL = """
INSERT OR REPLACE INTO products
    (product_id, name, category, color, brand, occasion, description,
     price_eur, sizes, in_stock)
VALUES
    (:product_id, :name, :category, :color, :brand, :occasion, :description,
     :price_eur, :sizes, :in_stock)
"""


# ---------------------------------------------------------------------------
# Step 1: load
# ---------------------------------------------------------------------------


def load_sample(path: str | Path) -> list[dict[str, Any]]:
    """Read the sample CSV and return rows as plain dicts."""
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------------------
# Step 2: clean
# ---------------------------------------------------------------------------


def clean(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Normalise and validate rows; return (kept, n_dropped).

    Drop rules (in order):
    1. Missing or empty `id` or `productDisplayName`.
    2. Duplicate `id` — keep the first occurrence.
    Normalise: color, category, and occasion to lowercase stripped strings.
    """
    kept: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    dropped = 0

    for row in rows:
        raw_id = (row.get("id") or "").strip()
        raw_name = (row.get("productDisplayName") or "").strip()

        if not raw_id or not raw_name:
            logger.debug("Dropping row — missing id or name: %s", row)
            dropped += 1
            continue

        if raw_id in seen_ids:
            logger.debug("Dropping row — duplicate id: %s", raw_id)
            dropped += 1
            continue

        seen_ids.add(raw_id)
        row = dict(row)  # don't mutate the input
        row["baseColour"] = (row.get("baseColour") or "").strip().lower()
        row["category"] = (row.get("category") or "").strip().lower()
        row["usage"] = (row.get("usage") or "").strip().lower()
        kept.append(row)

    if dropped:
        logger.info("clean: dropped %d rows (missing fields or duplicates)", dropped)

    return kept, dropped


# ---------------------------------------------------------------------------
# Step 3: enrich
# ---------------------------------------------------------------------------


def _extract_brand(display_name: str) -> str:
    """Use the first word of the display name as the brand.

    The Kaggle display names follow the pattern '<Brand> <descriptor> <product>',
    e.g. 'UCB Women Yellow T-shirt' → brand 'UCB'.
    """
    return display_name.split()[0] if display_name else "Unknown"


def _map_occasion(usage: str) -> str:
    return USAGE_TO_OCCASION.get(usage, "casual")


def enrich(rows: list[dict[str, Any]], seed: int = 42) -> list[dict[str, Any]]:
    """Add synthesised fields and return schema-complete product dicts.

    Synthesised fields use a seeded stdlib random.Random so output is
    deterministic. The seed is consumed in row order, matching load order,
    so the same CSV always produces the same enriched rows.
    """
    rng = random.Random(seed)
    result: list[dict[str, Any]] = []

    for row in rows:
        raw_id = row["id"].strip()
        product_id = f"PROD-{int(raw_id):05d}"
        name = row["productDisplayName"].strip()
        category = row["category"]
        color = row["baseColour"]
        occasion = _map_occasion(row["usage"])
        brand = _extract_brand(name)
        description = (
            f"{name}. {category.capitalize()} in {color}. Suitable for {occasion} occasions."
        )

        lo, hi = PRICE_RANGE.get(category, (20.0, 100.0))
        price_eur = round(rng.uniform(lo, hi), 2)

        size_list = SIZES_BY_CATEGORY.get(category, ["ONE"])
        stock_map = {s: rng.randint(0, 14) for s in size_list}
        in_stock = 1 if any(v > 0 for v in stock_map.values()) else 0

        result.append(
            {
                "product_id": product_id,
                "name": name,
                "category": category,
                "color": color,
                "brand": brand,
                "occasion": occasion,
                "description": description,
                "price_eur": price_eur,
                "sizes": json.dumps(stock_map),
                "in_stock": in_stock,
            }
        )

    return result


# ---------------------------------------------------------------------------
# Step 4: write to SQLite
# ---------------------------------------------------------------------------


def write_sqlite(rows: list[dict[str, Any]], db_path: str | Path) -> None:
    """Create the products table, insert rows, and rebuild the FTS index.

    Idempotent: uses CREATE TABLE IF NOT EXISTS and INSERT OR REPLACE,
    so running twice produces the same database with no duplicates or errors.
    The FTS5 index is rebuilt after every write so keyword search stays in sync.
    """
    from stylescout.catalog.search import build_fts_index

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.executemany(INSERT_SQL, rows)
        conn.commit()

    build_fts_index(db_path)
    logger.info("write_sqlite: wrote %d rows to %s", len(rows), db_path)


# ---------------------------------------------------------------------------
# Step 5: orchestrator
# ---------------------------------------------------------------------------


def build_catalog(
    sample_path: str | Path = "data/sample_styles.csv",
    db_path: str | Path = "data/catalog.db",
) -> dict[str, int]:
    """Run the full load → clean → enrich → write pipeline.

    Returns a summary dict with keys: loaded, dropped, written.
    """
    raw = load_sample(sample_path)
    cleaned, dropped = clean(raw)
    enriched = enrich(cleaned)
    write_sqlite(enriched, db_path)

    summary = {"loaded": len(raw), "dropped": dropped, "written": len(enriched)}
    logger.info("build_catalog: %s", summary)
    return summary
