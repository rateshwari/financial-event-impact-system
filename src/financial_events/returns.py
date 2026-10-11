import pandas as pd


def calculate_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate close-to-close simple returns per ticker."""
    required = {"date", "ticker", "close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = prices.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    df["return"] = df.groupby("ticker")["close"].pct_change()
    return df
