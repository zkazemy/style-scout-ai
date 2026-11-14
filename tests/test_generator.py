import json

import pandas as pd
import pytest

from stylescout.catalog.generator import generate


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    return generate()


def test_row_count(catalog: pd.DataFrame) -> None:
    assert len(catalog) == 2_000


def test_no_null_product_ids(catalog: pd.DataFrame) -> None:
    assert catalog["product_id"].notna().all()
    assert (catalog["product_id"] != "").all()


def test_price_positive(catalog: pd.DataFrame) -> None:
    assert (catalog["price_eur"] > 0).all()


def test_stock_non_negative(catalog: pd.DataFrame) -> None:
    for sizes_json in catalog["sizes"]:
        stock = json.loads(sizes_json)
        for qty in stock.values():
            assert qty >= 0
