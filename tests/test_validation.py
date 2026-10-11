import pandas as pd
import pytest
from financial_events.validation import load_news_csv, load_prices_csv


def test_news_deduplicates_ids(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame([
        {"news_id":"n1","published_at":"2025-01-01T10:00:00Z","headline":"First","source":"test"},
        {"news_id":"n1","published_at":"2025-01-01T10:00:00Z","headline":"Duplicate","source":"test"},
    ]).to_csv(path, index=False)
    result = load_news_csv(str(path))
    assert len(result) == 1
    assert result.iloc[0]["headline"] == "First"


def test_invalid_timestamp_fails(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame([{"news_id":"n1","published_at":"bad","headline":"x","source":"test"}]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="published_at"):
        load_news_csv(str(path))


def test_duplicate_prices_fail(tmp_path):
    path = tmp_path / "prices.csv"
    pd.DataFrame([
        {"date":"2025-01-01","ticker":"ABC","close":10},
        {"date":"2025-01-01","ticker":"ABC","close":11},
    ]).to_csv(path,index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_prices_csv(str(path))
