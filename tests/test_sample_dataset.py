import pandas as pd
import pytest

from stylescout.catalog.sample_dataset import (
    SAMPLE_SIZE,
    SEED,
    clean,
    load_raw,
    stratified_sample,
)


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    return stratified_sample(clean(load_raw()))


def test_sample_size(sample: pd.DataFrame) -> None:
    assert len(sample) == SAMPLE_SIZE


def test_product_ids_unique(sample: pd.DataFrame) -> None:
    assert sample["id"].nunique() == len(sample)


def test_same_seed_reproducible() -> None:
    df = clean(load_raw())
    s1 = stratified_sample(df, seed=SEED)
    s2 = stratified_sample(df, seed=SEED)
    pd.testing.assert_frame_equal(s1.reset_index(drop=True), s2.reset_index(drop=True))
