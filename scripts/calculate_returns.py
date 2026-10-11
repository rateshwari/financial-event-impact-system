from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/prices_clean.csv")
OUTPUT = Path("data/processed/daily_returns.csv")


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"Missing {INPUT}. Complete Step 2E first."
        )

    prices = pd.read_csv(INPUT, parse_dates=["date"])

    required = {"date", "ticker", "close"}
    if not required.issubset(prices.columns):
        raise ValueError(f"Required columns: {sorted(required)}")

    if prices.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker pairs found.")

    if prices["close"].isna().any() or (prices["close"] <= 0).any():
        raise ValueError("Invalid closing prices found.")

    prices = prices.sort_values(["ticker", "date"]).copy()

    # Calculate returns independently for each symbol.
    prices["daily_return"] = (
        prices.groupby("ticker")["close"].pct_change(fill_method=None)
    )

    # The first observation per symbol has no previous price.
    # Exclude it from the returns dataset rather than filling it with zero.
    returns = prices.dropna(subset=["daily_return"]).copy()

    if not returns["daily_return"].notna().all():
        raise ValueError("Missing returns remain.")

    # All symbols must share the same return dates.
    date_sets = {
        ticker: set(group["date"])
        for ticker, group in returns.groupby("ticker")
    }
    benchmark_dates = date_sets.get("^NSEI")

    if benchmark_dates is None:
        raise ValueError("NIFTY 50 benchmark is missing.")

    for ticker, dates in date_sets.items():
        if dates != benchmark_dates:
            raise ValueError(
                f"Return-date mismatch for {ticker}: "
                f"{len(dates ^ benchmark_dates)} differing dates."
            )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    returns[
        ["date", "ticker", "close", "daily_return"]
    ].to_csv(OUTPUT, index=False, date_format="%Y-%m-%d")

    print("RETURN CALCULATION SUCCESSFUL")
    print("Input:", INPUT)
    print("Output:", OUTPUT)
    print("Total return observations:", len(returns))
    print("Symbols:", sorted(returns["ticker"].unique()))
    print("Trading dates per symbol:", len(benchmark_dates))
    print(
        "Duplicate date/ticker pairs:",
        returns.duplicated(["date", "ticker"]).sum(),
    )

    print("\nReturn summary (decimal units):")
    print(
        returns.groupby("ticker")["daily_return"]
        .agg(["count", "mean", "std", "min", "max"])
        .to_string(float_format=lambda x: f"{x:.6f}")
    )

    print("\nLargest absolute daily returns:")
    largest = returns.loc[
        returns.groupby("ticker")["daily_return"]
        .apply(lambda x: x.abs().nlargest(2).index)
        .explode()
        .dropna()
        .astype(int)
    ]
    print(
        largest.assign(
            return_pct=largest["daily_return"] * 100
        )[["date", "ticker", "return_pct"]]
        .sort_values("return_pct", key=lambda x: x.abs(), ascending=False)
        .to_string(index=False, formatters={"return_pct": "{:.2f}%".format})
    )


if __name__ == "__main__":
    main()
