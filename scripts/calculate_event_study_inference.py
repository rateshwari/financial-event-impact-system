import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import t as student_t

RETURNS_FILE = Path("data/processed/daily_returns.csv")
EVENTS_FILE = Path("data/processed/event_registry_updated.csv")
OUTPUT_FILE = Path("data/processed/event_study_inference.csv")

BENCHMARK = "^NSEI"
STOCKS = [
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "RELIANCE.NS",
    "TCS.NS",
]

ESTIMATION_START = -120
ESTIMATION_END = -21

WINDOWS = {
    "day_0": (0, 0),
    "minus1_plus1": (-1, 1),
    "minus3_plus3": (-3, 3),
}


def main():
    returns = pd.read_csv(RETURNS_FILE, parse_dates=["date"])
    events = pd.read_csv(EVENTS_FILE, parse_dates=["event_date"])

    returns["date"] = returns["date"].dt.normalize()
    events["event_date"] = events["event_date"].dt.normalize()

    wide = returns.pivot(
        index="date", columns="ticker", values="daily_return"
    ).sort_index()

    required = [BENCHMARK] + STOCKS

    if wide[required].isna().any().any():
        raise ValueError("Missing required return observations.")

    dates = list(wide.index)
    date_to_index = {date: i for i, date in enumerate(dates)}

    primary_events = events[
        (events["event_category"] == "market_wide")
        & (events["event_status"] == "provisional")
    ]

    results = []

    for _, event in primary_events.iterrows():
        event_id = event["event_id"]
        event_date = event["event_date"]

        if event_date not in date_to_index:
            print(f"Skipping {event_id}: no matching trading date.")
            continue

        event_idx = date_to_index[event_date]
        est_start = event_idx + ESTIMATION_START
        est_end = event_idx + ESTIMATION_END

        if est_start < 0 or est_end >= len(dates):
            print(f"Skipping {event_id}: insufficient estimation history.")
            continue

        est_dates = dates[est_start:est_end + 1]
        est = wide.loc[est_dates, required]

        for stock in STOCKS:
            market_est = est[BENCHMARK].to_numpy()
            stock_est = est[stock].to_numpy()

            X_est = np.column_stack([
                np.ones(len(market_est)),
                market_est,
            ])

            coefficients = np.linalg.lstsq(
                X_est, stock_est, rcond=None
            )[0]

            residuals = stock_est - X_est @ coefficients
            n = len(stock_est)
            df = n - X_est.shape[1]

            if df <= 0:
                continue

            residual_variance = (
                residuals @ residuals
            ) / df

            xtx_inverse = np.linalg.inv(X_est.T @ X_est)

            for window_name, (left, right) in WINDOWS.items():
                start_idx = event_idx + left
                end_idx = event_idx + right

                if start_idx < 0 or end_idx >= len(dates):
                    continue

                event_dates = dates[start_idx:end_idx + 1]
                event_data = wide.loc[event_dates, [stock, BENCHMARK]]

                market_event = event_data[BENCHMARK].to_numpy()

                X_event = np.column_stack([
                    np.ones(len(market_event)),
                    market_event,
                ])

                actual = event_data[stock].to_numpy()
                expected = X_event @ coefficients
                abnormal = actual - expected
                car = abnormal.sum()

                # Prediction-error variance for the sum of abnormal
                # returns, including estimation-coefficient uncertainty.
                summed_design = X_event.sum(axis=0)

                variance_car = residual_variance * (
                    len(event_dates)
                    + summed_design @ xtx_inverse @ summed_design
                )

                se_car = np.sqrt(variance_car)
                t_stat = car / se_car if se_car > 0 else np.nan
                p_value = (
                    2 * student_t.sf(abs(t_stat), df)
                    if np.isfinite(t_stat)
                    else np.nan
                )

                results.append({
                    "event_id": event_id,
                    "event_date": event_date.date(),
                    "ticker": stock,
                    "window": window_name,
                    "window_start": event_dates[0].date(),
                    "window_end": event_dates[-1].date(),
                    "estimation_observations": n,
                    "car": car,
                    "car_percent_approx": car * 100,
                    "se_car": se_car,
                    "t_statistic": t_stat,
                    "degrees_freedom": df,
                    "p_value_two_sided": p_value,
                    "significant_5pct_uncorrected": (
                        p_value < 0.05 if np.isfinite(p_value) else False
                    ),
                    "inference_assumption": (
                        "Classical market-model prediction error; "
                        "independent homoskedastic residuals"
                    ),
                })

    output = pd.DataFrame(results)

    if output.empty:
        raise ValueError("No inference results were produced.")

    output.to_csv(OUTPUT_FILE, index=False)

    print("Saved:", OUTPUT_FILE)
    print("Rows:", len(output))
    print("\nInference results:")
    print(
        output[
            [
                "event_id",
                "ticker",
                "window",
                "car_percent_approx",
                "se_car",
                "t_statistic",
                "p_value_two_sided",
                "significant_5pct_uncorrected",
            ]
        ].round(5).to_string(index=False)
    )

    print(
        "\nCaution: p-values are exploratory, unadjusted for multiple "
        "testing, and conditional on the event dates and model assumptions."
    )


if __name__ == "__main__":
    main()
