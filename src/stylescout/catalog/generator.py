"""Generates a synthetic product catalog and saves it to CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NUM_PRODUCTS = 2_000

CATEGORIES = ["dress", "shirt", "trousers", "shoes", "jacket", "accessory"]
COLORS = ["black", "white", "blue", "red", "green", "grey", "beige", "brown", "pink", "navy"]
OCCASIONS = ["casual", "formal", "wedding", "sport", "evening", "outdoor"]
BRANDS = [
    "Zara",
    "H&M",
    "Mango",
    "Uniqlo",
    "COS",
    "Arket",
    "Weekday",
    "& Other Stories",
    "Monki",
    "Pull&Bear",
]
SIZES_BY_CATEGORY: dict[str, list[str]] = {
    "dress": ["XS", "S", "M", "L", "XL"],
    "shirt": ["XS", "S", "M", "L", "XL", "XXL"],
    "trousers": ["28", "30", "32", "34", "36", "38"],
    "shoes": ["36", "37", "38", "39", "40", "41", "42", "43"],
    "jacket": ["XS", "S", "M", "L", "XL"],
    "accessory": ["ONE"],
}
PRICE_RANGE: dict[str, tuple[float, float]] = {
    "dress": (30.0, 200.0),
    "shirt": (20.0, 120.0),
    "trousers": (25.0, 150.0),
    "shoes": (40.0, 250.0),
    "jacket": (50.0, 350.0),
    "accessory": (10.0, 80.0),
}
ADJECTIVES = [
    "Classic",
    "Slim-fit",
    "Relaxed",
    "Oversized",
    "Cropped",
    "Flowy",
    "Structured",
    "Wrap",
    "Ribbed",
    "Printed",
    "Striped",
    "Checked",
]


def _make_description(name: str, color: str, category: str, occasion: str) -> str:
    return (
        f"{name}. A {color} {category} suitable for {occasion} wear. "
        "Made from quality materials with attention to detail and comfort."
    )


def _make_sizes(rng: np.random.Generator, category: str) -> tuple[str, int]:
    """Return (sizes JSON, in_stock flag)."""
    size_list = SIZES_BY_CATEGORY[category]
    stock_per_size = {s: int(rng.integers(0, 15)) for s in size_list}
    in_stock = 1 if any(v > 0 for v in stock_per_size.values()) else 0
    return json.dumps(stock_per_size), in_stock


def generate(num_products: int = NUM_PRODUCTS, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = rng.choice(CATEGORIES, size=num_products)
    colors = rng.choice(COLORS, size=num_products)
    occasions = rng.choice(OCCASIONS, size=num_products)
    brands = rng.choice(BRANDS, size=num_products)
    adjectives = rng.choice(ADJECTIVES, size=num_products)

    records = []
    for i in range(num_products):
        cat = categories[i]
        color = colors[i]
        occasion = occasions[i]
        brand = str(brands[i])
        adj = adjectives[i]

        product_id = f"PROD-{i + 1:05d}"
        name = f"{adj} {color.capitalize()} {cat.capitalize()}"
        lo, hi = PRICE_RANGE[cat]
        price = round(float(rng.uniform(lo, hi)), 2)
        sizes_json, in_stock = _make_sizes(rng, cat)
        description = _make_description(name, color, cat, occasion)

        records.append(
            {
                "product_id": product_id,
                "name": name,
                "category": cat,
                "color": color,
                "brand": brand,
                "occasion": occasion,
                "description": description,
                "price_eur": price,
                "sizes": sizes_json,
                "in_stock": in_stock,
            }
        )

    return pd.DataFrame(records)


def save(df: pd.DataFrame, path: str | Path = "data/synthetic_catalog.csv") -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


def main() -> None:
    df = generate()
    save(df)
    print(f"Wrote {len(df)} products to data/synthetic_catalog.csv")


if __name__ == "__main__":
    main()
