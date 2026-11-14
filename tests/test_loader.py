"""Tests for catalog/loader.py using a small in-memory fixture."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from stylescout.catalog.loader import (
    build_catalog,
    clean,
    enrich,
    write_sqlite,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

FIXTURE_ROWS = [
    {
        "id": "1001",
        "gender": "Women",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Tops",
        "baseColour": "Blue",
        "season": "Summer",
        "year": "2012.0",
        "usage": "Casual",
        "productDisplayName": "BrandA Women Blue Top",
        "category": "shirt",
    },
    {
        "id": "1002",
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Bottomwear",
        "articleType": "Jeans",
        "baseColour": "Black",
        "season": "Fall",
        "year": "2011.0",
        "usage": "Casual",
        "productDisplayName": "BrandB Men Black Jeans",
        "category": "trousers",
    },
    {
        "id": "1003",
        "gender": "Women",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Dresses",
        "baseColour": "Red",
        "season": "Summer",
        "year": "2013.0",
        "usage": "Formal",
        "productDisplayName": "BrandC Women Red Dress",
        "category": "dress",
    },
    {
        "id": "1004",
        "gender": "Men",
        "masterCategory": "Footwear",
        "subCategory": "Shoes",
        "articleType": "Casual Shoes",
        "baseColour": "Brown",
        "season": "Fall",
        "year": "2010.0",
        "usage": "Casual",
        "productDisplayName": "BrandD Men Brown Shoes",
        "category": "shoes",
    },
    # Row with missing productDisplayName — should be dropped
    {
        "id": "1005",
        "gender": "Men",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Shirts",
        "baseColour": "White",
        "season": "Summer",
        "year": "2011.0",
        "usage": "Casual",
        "productDisplayName": "",
        "category": "shirt",
    },
    # Duplicate id 1001 — should be dropped
    {
        "id": "1001",
        "gender": "Women",
        "masterCategory": "Apparel",
        "subCategory": "Topwear",
        "articleType": "Tops",
        "baseColour": "Green",
        "season": "Summer",
        "year": "2012.0",
        "usage": "Casual",
        "productDisplayName": "BrandA Women Green Top",
        "category": "shirt",
    },
]

FIELDNAMES = list(FIXTURE_ROWS[0].keys())


def _write_fixture_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture()
def fixture_csv(tmp_path: Path) -> Path:
    p = tmp_path / "fixture.csv"
    _write_fixture_csv(FIXTURE_ROWS, p)
    return p


@pytest.fixture()
def cleaned_rows() -> list[dict]:
    kept, _ = clean(FIXTURE_ROWS)
    return kept


@pytest.fixture()
def enriched_rows(cleaned_rows: list[dict]) -> list[dict]:
    return enrich(cleaned_rows)


# ---------------------------------------------------------------------------
# Tests: clean
# ---------------------------------------------------------------------------


def test_clean_drops_missing_name() -> None:
    kept, dropped = clean(FIXTURE_ROWS)
    assert dropped == 2  # empty name + duplicate id


def test_clean_row_count(cleaned_rows: list[dict]) -> None:
    assert len(cleaned_rows) == 4


def test_clean_no_null_product_ids(cleaned_rows: list[dict]) -> None:
    assert all(r["id"].strip() for r in cleaned_rows)


def test_clean_normalises_color(cleaned_rows: list[dict]) -> None:
    for row in cleaned_rows:
        assert row["baseColour"] == row["baseColour"].lower()


def test_clean_normalises_category(cleaned_rows: list[dict]) -> None:
    for row in cleaned_rows:
        assert row["category"] == row["category"].lower()


# ---------------------------------------------------------------------------
# Tests: enrich
# ---------------------------------------------------------------------------


def test_enrich_price_positive(enriched_rows: list[dict]) -> None:
    assert all(r["price_eur"] > 0 for r in enriched_rows)


def test_enrich_stock_non_negative(enriched_rows: list[dict]) -> None:
    for row in enriched_rows:
        for qty in json.loads(row["sizes"]).values():
            assert qty >= 0


def test_enrich_in_stock_consistent(enriched_rows: list[dict]) -> None:
    for row in enriched_rows:
        stock = json.loads(row["sizes"])
        expected = 1 if any(v > 0 for v in stock.values()) else 0
        assert row["in_stock"] == expected


def test_enrich_same_seed_reproducible(cleaned_rows: list[dict]) -> None:
    a = enrich(cleaned_rows, seed=42)
    b = enrich(cleaned_rows, seed=42)
    assert [r["price_eur"] for r in a] == [r["price_eur"] for r in b]
    assert [r["sizes"] for r in a] == [r["sizes"] for r in b]


def test_enrich_different_seeds_differ(cleaned_rows: list[dict]) -> None:
    a = enrich(cleaned_rows, seed=42)
    b = enrich(cleaned_rows, seed=99)
    # With different seeds the price lists should not be identical
    assert [r["price_eur"] for r in a] != [r["price_eur"] for r in b]


# ---------------------------------------------------------------------------
# Tests: write_sqlite
# ---------------------------------------------------------------------------


def test_write_sqlite_idempotent(enriched_rows: list[dict], tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    write_sqlite(enriched_rows, db)
    write_sqlite(enriched_rows, db)  # second write must not raise or duplicate

    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert count == len(enriched_rows)


# ---------------------------------------------------------------------------
# Tests: build_catalog end-to-end
# ---------------------------------------------------------------------------


def test_build_catalog_summary(fixture_csv: Path, tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    summary = build_catalog(sample_path=fixture_csv, db_path=db)
    assert summary["loaded"] == len(FIXTURE_ROWS)
    assert summary["dropped"] == 2
    assert summary["written"] == 4


def test_build_catalog_twice_unchanged(fixture_csv: Path, tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    build_catalog(sample_path=fixture_csv, db_path=db)
    build_catalog(sample_path=fixture_csv, db_path=db)

    with sqlite3.connect(db) as conn:
        count = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert count == 4


def test_build_catalog_db_exists(fixture_csv: Path, tmp_path: Path) -> None:
    db = tmp_path / "catalog.db"
    build_catalog(sample_path=fixture_csv, db_path=db)
    assert db.exists()
