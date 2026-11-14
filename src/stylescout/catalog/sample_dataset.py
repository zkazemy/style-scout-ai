"""Draw a reproducible stratified sample from the raw Kaggle styles.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/styles.csv")
OUT_CSV = Path("data/sample_styles.csv")
SAMPLE_SIZE = 3_000
SEED = 42

# Map Kaggle articleType → our 6-category fixed list.
# Types not in this map are excluded from the sample.
ARTICLE_TYPE_TO_CATEGORY: dict[str, str] = {
    # dress
    "Dresses": "dress",
    "Nightdress": "dress",
    "Sarees": "dress",
    "Kurtis": "dress",
    "Skirts": "dress",
    "Night suits": "dress",
    "Tunics": "dress",
    "Rompers": "dress",
    "Jumpsuit": "dress",
    "Baby Dolls": "dress",
    "Lehenga Choli": "dress",
    # shirt
    "Tshirts": "shirt",
    "Shirts": "shirt",
    "Tops": "shirt",
    "Kurtas": "shirt",
    "Sweatshirts": "shirt",
    "Sweaters": "shirt",
    "Innerwear Vests": "shirt",
    "Camisoles": "shirt",
    "Lounge Tshirts": "shirt",
    "Shrug": "shirt",
    # trousers
    "Jeans": "trousers",
    "Trousers": "trousers",
    "Shorts": "trousers",
    "Track Pants": "trousers",
    "Capris": "trousers",
    "Leggings": "trousers",
    "Lounge Pants": "trousers",
    "Lounge Shorts": "trousers",
    "Jeggings": "trousers",
    "Patiala": "trousers",
    "Salwar": "trousers",
    "Churidar": "trousers",
    "Tights": "trousers",
    "Stockings": "trousers",
    "Boxers": "trousers",
    "Trunk": "trousers",
    "Briefs": "trousers",
    # shoes
    "Casual Shoes": "shoes",
    "Sports Shoes": "shoes",
    "Heels": "shoes",
    "Flip Flops": "shoes",
    "Sandals": "shoes",
    "Formal Shoes": "shoes",
    "Flats": "shoes",
    "Sports Sandals": "shoes",
    "Booties": "shoes",
    # jacket
    "Jackets": "jacket",
    "Rain Jacket": "jacket",
    "Blazers": "jacket",
    "Waistcoat": "jacket",
    "Nehru Jackets": "jacket",
    "Bath Robe": "jacket",
    "Robe": "jacket",
    # accessory
    "Watches": "accessory",
    "Handbags": "accessory",
    "Sunglasses": "accessory",
    "Wallets": "accessory",
    "Belts": "accessory",
    "Backpacks": "accessory",
    "Caps": "accessory",
    "Ties": "accessory",
    "Earrings": "accessory",
    "Clutches": "accessory",
    "Pendant": "accessory",
    "Necklace and Chains": "accessory",
    "Scarves": "accessory",
    "Ring": "accessory",
    "Dupatta": "accessory",
    "Cufflinks": "accessory",
    "Stoles": "accessory",
    "Bangle": "accessory",
    "Bracelet": "accessory",
    "Jewellery Set": "accessory",
    "Duffel Bag": "accessory",
    "Laptop Bag": "accessory",
    "Mobile Pouch": "accessory",
    "Messenger Bag": "accessory",
    "Mufflers": "accessory",
    "Gloves": "accessory",
    "Socks": "accessory",
    "Headband": "accessory",
    "Wristbands": "accessory",
    "Suspenders": "accessory",
    "Waist Pouch": "accessory",
    "Key chain": "accessory",
    "Umbrellas": "accessory",
    "Rucksacks": "accessory",
    "Trolley Bag": "accessory",
    "Swimwear": "accessory",
}


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    return pd.read_csv(path, on_bad_lines="skip")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be mapped to our schema."""
    df = df.dropna(subset=["productDisplayName", "baseColour"])
    df = df[df["articleType"].isin(ARTICLE_TYPE_TO_CATEGORY)]
    df = df.copy()
    df["category"] = df["articleType"].map(ARTICLE_TYPE_TO_CATEGORY)
    return df


def stratified_sample(df: pd.DataFrame, n: int = SAMPLE_SIZE, seed: int = SEED) -> pd.DataFrame:
    """
    Sample `n` rows balanced across categories.

    Each category gets a quota of n // n_categories. Categories with fewer
    rows than their quota contribute all their rows; the shortfall is
    redistributed proportionally among the remaining categories, capped so
    the total stays at exactly `n`.
    """
    categories = df["category"].unique().tolist()
    n_cats = len(categories)
    base_quota = n // n_cats

    # First pass: allocate quotas
    counts = df["category"].value_counts()
    quotas: dict[str, int] = {}
    shortfall = 0
    for cat in categories:
        available = int(counts.get(cat, 0))
        if available <= base_quota:
            quotas[cat] = available
            shortfall += base_quota - available
        else:
            quotas[cat] = base_quota

    # Second pass: redistribute shortfall to categories that have surplus
    surplus_cats = [c for c in categories if int(counts.get(c, 0)) > base_quota]
    if surplus_cats and shortfall > 0:
        extra = shortfall // len(surplus_cats)
        remainder = shortfall % len(surplus_cats)
        for i, cat in enumerate(surplus_cats):
            additional = extra + (1 if i < remainder else 0)
            cap = int(counts.get(cat, 0)) - base_quota
            quotas[cat] += min(additional, cap)

    # Draw samples
    parts: list[pd.DataFrame] = []
    for cat in categories:
        pool = df[df["category"] == cat]
        k = min(quotas[cat], len(pool))
        parts.append(pool.sample(n=k, random_state=seed))

    result = pd.concat(parts, ignore_index=True)

    # If we're still short due to rounding, top up from the largest category
    deficit = n - len(result)
    if deficit > 0:
        already_sampled_ids = set(result["id"])
        leftovers = df[~df["id"].isin(already_sampled_ids)]
        result = pd.concat(
            [result, leftovers.sample(n=deficit, random_state=seed)],
            ignore_index=True,
        )

    return result.head(n)


def save(df: pd.DataFrame, path: Path = OUT_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    raw = load_raw()
    cleaned = clean(raw)
    sample = stratified_sample(cleaned)
    save(sample)
    print(f"Wrote {len(sample)} rows to {OUT_CSV}")
    print("Category distribution:")
    print(sample["category"].value_counts().to_string())


if __name__ == "__main__":
    main()
