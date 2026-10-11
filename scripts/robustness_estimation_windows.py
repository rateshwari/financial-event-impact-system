import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as student_t

RETURNS_FILE = Path("data/processed/daily_returns.csv")
OUTPUT_FILE = Path("data/processed/tcs_estimation_window_robustness.csv")

BENCHMARK = "^NSEI"
STOCK = "TCS.NS"
EVENT_DATE = pd.Timestamp("2024-06-04")

# End each estimation window 21 trading days before the event.
WINDOWS = {
    "baseline_-120_to_-21": (-120, -21),
    "alternative_-150_to_-21": (-150, -21),
    "alternative_-250_to_-21": (-250, -21),
}


def main():
    df = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    df["date"] = df["date"].dt.normalize()

    wide = df.pivot(
        index="date",
        columns="ticker",
        values="daily_return",
    ).sort_index()

    required = [STOCK, BENCHMARK]
    if wide[required].isna().any().any():
        raise ValueError("Missing stock or benchmark returns.")

    dates = list(wide.index)
    date_to_idx = {date: i for i, date in enumerate(dates)}

    if EVENT_DATE not in date_to_idx:
        raise ValueError("Event date is absent from the trading calendar.")

    event_idx = date_to_idx[EVENT_DATE]
    event_row = wide.loc[EVENT_DATE]
    market_event = float(event_row[BENCHMARK])
    actual_event_return = float(event_row[STOCK])

    results = []

    for name, (start_offset, end_offset) in WINDOWS.items():
        start_idx = event_idx + start_offset
        end_idx = event_idx + end_offset

        if start_idx < 0 or end_idx >= len(dates):
            raise ValueError(f"Insufficient data for {name}.")

        est_dates = dates[start_idx:end_idx + 1]
        est = wide.loc[est_dates, required]

        market = est[BENCHMARK].to_numpy()
        stock = est[STOCK].to_numpy()

        X = np.column_stack([np.ones(len(market)), market])
        alpha, beta = np.linalg.lstsq(X, stock, rcond=None)[0]

        residuals = stock - X @ np.array([alpha, beta])
        n = len(stock)
        degrees_freedom = n - 2
        residual_variance = float(residuals @ residuals / degrees_freedom)

        x0 = np.array([1.0, market_event])
        expected = float(x0 @ np.array([alpha, beta]))
        abnormal = actual_event_return - expected

        xtx_inverse = np.linalg.inv(X.T @ X)
        prediction_variance = residual_variance * (
            1 + x0 @ xtx_inverse @ x0
        )
        se = float(np.sqrt(prediction_variance))
        t_stat = abnormal / se
        p_value = float(
            2 * student_t.sf(abs(t_stat), degrees_freedom)
        )

        results.append({
            "estimation_window": name,
            "estimation_start": est_dates[0].date(),
            "estimation_end": est_dates[-1].date(),
            "observations": n,
            "alpha_daily": alpha,
            "beta_market": beta,
            "residual_std_daily": np.sqrt(residual_variance),
            "actual_return": actual_event_return,
            "expected_return": expected,
            "abnormal_return": abnormal,
            "abnormal_return_percent": abnormal * 100,
            "standard_error": se,
            "t_statistic": t_stat,
            "degrees_freedom": degrees_freedom,
            "p_value_uncorrected": p_value,
        })

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print(out[[
        "estimation_window",
        "observations",
        "beta_market",
        "abnormal_return_percent",
        "p_value_uncorrected",
    ]].round(5).to_string(index=False))


if __name__ == "__main__":
    main()
