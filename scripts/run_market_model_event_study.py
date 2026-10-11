import pandas as pd
import numpy as np
from pathlib import Path

RETURNS_FILE = Path("data/processed/daily_returns.csv")
EVENTS_FILE = Path("data/processed/event_registry_updated.csv")
OUTPUT_DIR = Path("data/processed")

BENCHMARK = "^NSEI"
STOCKS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "TCS.NS",
]

# Estimation window relative to event trading-day index.
ESTIMATION_START = -120
ESTIMATION_END = -21

EVENT_WINDOWS = {
    "day_0": (0, 0),
    "minus1_plus1": (-1, 1),
    "minus3_plus3": (-3, 3),
}


def main():
    returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    events = pd.read_csv(EVENTS_FILE, parse_dates=["event_date"])

    returns["date"] = returns["date"].dt.normalize()
    events["event_date"] = events["event_date"].dt.normalize()

    if returns.duplicated(["date", "ticker"]).any():
        raise ValueError("Duplicate date/ticker observations found.")

    wide = returns.pivot(
        index="date",
        columns="ticker",
        values="daily_return",
    ).sort_index()

    required = [BENCHMARK] + STOCKS
    missing = sorted(set(required) - set(wide.columns))
    if missing:
        raise ValueError(f"Missing required tickers: {missing}")

    if wide[required].isna().any().any():
        raise ValueError(
            "Missing returns found in required series; "
            "resolve missing observations before estimating."
        )

    dates = list(wide.index)
    date_to_index = {date: i for i, date in enumerate(dates)}

    # Retain market-wide events only for the primary five-stock analysis.
    primary_events = events[
        (events["event_category"] == "market_wide")
        & (events["event_status"] == "provisional")
    ].copy()

    estimation_records = []
    abnormal_records = []
    car_records = []
    event_index_records = []

    for _, event in primary_events.iterrows():
        event_date = event["event_date"]
        event_id = event["event_id"]

        if event_date not in date_to_index:
            print(f"Skipping {event_id}: event date is not a trading date.")
            continue

        event_idx = date_to_index[event_date]
        event_index_records.append({
            "event_id": event_id,
            "event_date": event_date.date(),
            "event_trading_index": event_idx,
        })

        est_start = event_idx + ESTIMATION_START
        est_end = event_idx + ESTIMATION_END

        if est_start < 0 or est_end >= len(dates):
            print(f"Skipping {event_id}: insufficient estimation history.")
            continue

        estimation_dates = dates[est_start:est_end + 1]
        estimation = wide.loc[estimation_dates, required].dropna()

        market_returns = estimation[BENCHMARK].to_numpy()

        for stock in STOCKS:
            stock_returns = estimation[stock].to_numpy()

            # OLS: stock_return = alpha + beta * market_return
            X = np.column_stack([
                np.ones(len(market_returns)),
                market_returns,
            ])
            alpha, beta = np.linalg.lstsq(
                X, stock_returns, rcond=None
            )[0]

            fitted = alpha + beta * market_returns
            residuals = stock_returns - fitted
            residual_std = residuals.std(ddof=2)

            estimation_records.append({
                "event_id": event_id,
                "event_date": event_date.date(),
                "ticker": stock,
                "estimation_observations": len(estimation),
                "alpha_daily": alpha,
                "beta_market": beta,
                "residual_std_daily": residual_std,
            })

            for window_name, (left, right) in EVENT_WINDOWS.items():
                start_idx = event_idx + left
                end_idx = event_idx + right

                if start_idx < 0 or end_idx >= len(dates):
                    continue

                window_dates = dates[start_idx:end_idx + 1]
                window = wide.loc[window_dates, [stock, BENCHMARK]]

                expected = alpha + beta * window[BENCHMARK]
                abnormal = window[stock] - expected

                for date, actual, expected_return, ar in zip(
                    window.index,
                    window[stock],
                    expected,
                    abnormal,
                ):
                    abnormal_records.append({
                        "event_id": event_id,
                        "event_date": event_date.date(),
                        "ticker": stock,
                        "window": window_name,
                        "date": date.date(),
                        "actual_return": actual,
                        "expected_return": expected_return,
                        "abnormal_return": ar,
                    })

                car_records.append({
                    "event_id": event_id,
                    "event_date": event_date.date(),
                    "ticker": stock,
                    "window": window_name,
                    "window_start": window_dates[0].date(),
                    "window_end": window_dates[-1].date(),
                    "trading_days": len(window_dates),
                    "car": abnormal.sum(),
                    "car_percent_approx": abnormal.sum() * 100,
                })

    estimation_df = pd.DataFrame(estimation_records)
    abnormal_df = pd.DataFrame(abnormal_records)
    car_df = pd.DataFrame(car_records)
    event_index_df = pd.DataFrame(event_index_records)

    # Mark overlapping event windows for the same stock and window type.
    if not car_df.empty:
        car_df["overlap_warning"] = False

        for window_name in EVENT_WINDOWS:
            subset = car_df[car_df["window"] == window_name]
            for ticker in STOCKS:
                rows = subset[subset["ticker"] == ticker]
                indices = rows.index.tolist()

                for i, idx_a in enumerate(indices):
                    a = car_df.loc[idx_a]
                    a_start = pd.Timestamp(a["window_start"])
                    a_end = pd.Timestamp(a["window_end"])

                    for idx_b in indices[i + 1:]:
                        b = car_df.loc[idx_b]
                        b_start = pd.Timestamp(b["window_start"])
                        b_end = pd.Timestamp(b["window_end"])

                        if max(a_start, b_start) <= min(a_end, b_end):
                            car_df.loc[idx_a, "overlap_warning"] = True
                            car_df.loc[idx_b, "overlap_warning"] = True

    outputs = {
        "event_study_estimation_parameters.csv": estimation_df,
        "event_study_abnormal_returns.csv": abnormal_df,
        "event_study_car_results.csv": car_df,
        "event_study_event_indices.csv": event_index_df,
    }

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"Saved {path} ({len(frame)} rows)")

    print("\nEstimation parameter summary:")
    if not estimation_df.empty:
        print(
            estimation_df[
                ["ticker", "estimation_observations",
                 "alpha_daily", "beta_market"]
            ].to_string(index=False)
        )

    print("\nCAR results:")
    if not car_df.empty:
        print(
            car_df[
                ["event_id", "ticker", "window",
                 "car", "overlap_warning"]
            ].to_string(index=False)
        )

    print(
        "\nInterpretation warning: CAR is not automatically statistically "
        "significant or causal. Statistical inference and event-timing "
        "validation are still required."
    )


if __name__ == "__main__":
    main()
