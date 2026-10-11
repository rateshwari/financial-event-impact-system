import pandas as pd

NEWS_REQUIRED = {"news_id", "published_at", "headline", "source"}
PRICE_REQUIRED = {"date", "ticker", "close"}


def load_news_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = NEWS_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"News CSV missing required columns: {sorted(missing)}")
    df["news_id"] = df["news_id"].astype("string").str.strip()
    df["headline"] = df["headline"].astype("string").str.strip()
    df["source"] = df["source"].astype("string").str.strip()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    if df["news_id"].isna().any() or (df["news_id"] == "").any():
        raise ValueError("news_id must be non-empty")
    if df["headline"].isna().any() or (df["headline"] == "").any():
        raise ValueError("headline must be non-empty")
    if df["published_at"].isna().any():
        raise ValueError("Some published_at values are invalid; use ISO-8601 timestamps")
    return df.drop_duplicates(subset=["news_id"], keep="first").reset_index(drop=True)


def load_prices_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = PRICE_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Prices CSV missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    df["ticker"] = df["ticker"].astype("string").str.strip().str.upper()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some price dates are invalid")
    if df["ticker"].isna().any() or (df["ticker"] == "").any():
        raise ValueError("ticker must be non-empty")
    if df["close"].isna().any() or (df["close"] <= 0).any():
        raise ValueError("close must be a positive numeric value")
    if df.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker price rows found")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)
