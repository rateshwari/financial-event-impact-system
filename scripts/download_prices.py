#!/usr/bin/env python3
"""Download daily adjusted OHLC data for a small Indian equity universe.

Yahoo Finance is a convenient research prototype source, not a guaranteed
institutional data feed. Respect its terms and validate downloads before use.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "^NSEI",  # NIFTY 50 market benchmark
]


def download_prices(tickers: list[str], start: str, end: str, output: Path) -> pd.DataFrame:
    # auto_adjust=True adjusts OHLC for corporate actions. The resulting Close
    # column is adjusted; do not label it raw close.
    frame = yf.download(
        tickers=tickers,
        start=start,
        end=end,  # yfinance end date is exclusive
        interval="1d",
        auto_adjust=True,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
        multi_level_index=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("No market data returned. Check tickers, dates, network, or provider limits.")

    # yfinance may return MultiIndex columns depending on version/ticker count.
    if isinstance(frame.columns, pd.MultiIndex):
        # Expected shape when grouping by ticker: first level ticker, second OHLCV.
        if "Close" in frame.columns.get_level_values(-1):
            close = frame.xs("Close", axis=1, level=-1)
        elif "Close" in frame.columns.get_level_values(0):
            close = frame.xs("Close", axis=1, level=0)
        else:
            raise RuntimeError(f"Could not find Close in returned columns: {frame.columns}")
    else:
        if "Close" not in frame.columns:
            raise RuntimeError(f"Could not find Close in returned columns: {list(frame.columns)}")
        # A single ticker returns a Series-like one-column frame.
        if len(tickers) == 1:
            close = frame[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            raise RuntimeError("Unexpected flat columns for multiple tickers; inspect yfinance output.")

    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.index.name = "date"
    long = close.rename_axis(columns="ticker").stack(future_stack=True).rename("close").reset_index()
    long["ticker"] = long["ticker"].astype(str).str.upper()
    long["close"] = pd.to_numeric(long["close"], errors="coerce")
    long = long.dropna(subset=["close"])
    long = long[long["close"] > 0]
    long = long.sort_values(["ticker", "date"]).reset_index(drop=True)

    if long.empty:
        raise RuntimeError("Downloaded frame had no valid closing prices.")
    if long.duplicated(["date", "ticker"]).any():
        raise RuntimeError("Provider returned duplicate date/ticker rows.")

    output.parent.mkdir(parents=True, exist_ok=True)
    # ISO date format; record source and adjustment convention in sidecar metadata.
    long.to_csv(output, index=False, date_format="%Y-%m-%d")
    metadata = output.with_suffix(".metadata.txt")
    metadata.write_text(
        "source=Yahoo Finance via yfinance\n"
        f"tickers={','.join(tickers)}\n"
        f"start_inclusive={start}\n"
        f"end_exclusive={end}\n"
        "interval=1d\n"
        "price_column=Close with auto_adjust=True (adjusted OHLC)\n"
        f"downloaded_at_utc={pd.Timestamp.now(tz='UTC').isoformat()}\n",
        encoding="utf-8",
    )
    return long


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2022-01-01", help="Inclusive start date YYYY-MM-DD")
    parser.add_argument(
        "--end",
        default=(date.today() - timedelta(days=1)).isoformat(),
        help="Exclusive end date YYYY-MM-DD (default: yesterday)",
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output", default="data/raw/prices.csv")
    args = parser.parse_args()

    data = download_prices(args.tickers, args.start, args.end, Path(args.output))
    print(f"Saved {len(data):,} price rows to {args.output}")
    print(f"Tickers: {', '.join(sorted(data['ticker'].unique()))}")
    print(f"Date range: {data['date'].min().date()} through {data['date'].max().date()}")
    print("Reminder: verify ticker coverage and adjusted-price convention before analysis.")


if __name__ == "__main__":
    main()
