
import pandas as pd
from pathlib import Path

BASE = Path("data/processed")
OUTPUT = BASE / "robustness_summary.csv"

FILES = {
    "adjusted_event_study": BASE / "event_study_inference_adjusted.csv",
    "placebo_dates": BASE / "tcs_placebo_test.csv",
    "extreme_market_placebo": BASE / "tcs_extreme_market_placebos.csv",
    "hc3_diagnostic": BASE / "tcs_hc3_inference.csv",
    "window_robustness": BASE / "tcs_estimation_window_robustness.csv",
}

records = []


def add_record(method, event_date, ticker, metric, value, interpretation):
    records.append({
        "method": method,
        "event_date": event_date,
        "ticker": ticker,
        "metric": metric,
        "value": value,
        "interpretation": interpretation,
    })


# 1. Existing multiple-testing-adjusted event-study results.
path = FILES["adjusted_event_study"]

if path.exists():
    df = pd.read_csv(path)
    print(f"\nAdjusted event-study columns: {list(df.columns)}")

    # Preserve the original result table's values in a normalized summary.
    for _, row in df.iterrows():
        event_date = row.get("event_date", row.get("date", ""))
        ticker = row.get("ticker", "")
        window = row.get("window", row.get("event_window", ""))
        event_label = f"{event_date} | {window}".strip(" |")

        for col in df.columns:
            if col in {
                "event_id", "event_date", "date", "ticker",
                "window", "event_window"
            }:
                continue

            value = row[col]
            if pd.isna(value):
                continue

            add_record(
                "adjusted_event_study",
                event_label,
                ticker,
                col,
                value,
                "Original adjusted event-study output; see source CSV."
            )
else:
    print("WARNING: adjusted event-study output not found:", path)


# 2. Broad historical placebo comparison.
path = FILES["placebo_dates"]

if path.exists():
    df = pd.read_csv(path)
    placebo = df[df["is_event_date"].astype(str).str.lower() == "false"].copy()
    event = df[df["is_event_date"].astype(str).str.lower() == "true"].copy()

    if not event.empty and not placebo.empty:
        event_ar = float(event.iloc[0]["abnormal_return"])
        count = int((placebo["abnormal_return"].abs() >= abs(event_ar)).sum())
        p_empirical = (count + 1) / (len(placebo) + 1)

        add_record(
            "broad_placebo",
            "2024-06-04",
            "TCS.NS",
            "event_abnormal_return",
            event_ar,
            "Market-model abnormal return, not actual stock return."
        )
        add_record(
            "broad_placebo",
            "2024-06-04",
            "TCS.NS",
            "eligible_placebo_dates",
            len(placebo),
            "Historical comparison dates after exclusion rules."
        )
        add_record(
            "broad_placebo",
            "2024-06-04",
            "TCS.NS",
            "empirical_tail_estimate",
            p_empirical,
            "Add-one historical tail estimate; not a randomized-test p-value."
        )


# 3. Extreme-market placebo comparison.
path = FILES["extreme_market_placebo"]

if path.exists():
    df = pd.read_csv(path)
    event = df[df["comparison_group"] == "event"]
    placebo = df[df["comparison_group"] == "extreme_market_placebo"]

    if not event.empty and not placebo.empty:
        event_ar = float(event.iloc[0]["abnormal_return"])
        count = int((placebo["abnormal_return"].abs() >= abs(event_ar)).sum())
        p_empirical = (count + 1) / (len(placebo) + 1)

        add_record(
            "extreme_market_placebo",
            "2024-06-04",
            "TCS.NS",
            "eligible_placebo_dates",
            len(placebo),
            "Only a small number qualified; interpret descriptively."
        )
        add_record(
            "extreme_market_placebo",
            "2024-06-04",
            "TCS.NS",
            "dates_at_least_as_extreme",
            count,
            "Count of eligible dates with absolute AR at least as large."
        )
        add_record(
            "extreme_market_placebo",
            "2024-06-04",
            "TCS.NS",
            "empirical_tail_estimate",
            p_empirical,
            "Small comparison sample; not strong inferential evidence."
        )


# 4. HC3 supplementary diagnostic.
path = FILES["hc3_diagnostic"]

if path.exists():
    df = pd.read_csv(path)
    if not df.empty:
        row = df.iloc[0]

        for col in [
            "abnormal_return",
            "robust_prediction_standard_error",
            "normal_approx_z_statistic",
            "normal_approx_two_sided_p",
        ]:
            if col in df.columns:
                add_record(
                    "HC3_supplementary_diagnostic",
                    str(row.get("event_date", "2024-06-04")),
                    str(row.get("ticker", "TCS.NS")),
                    col,
                    row[col],
                    "Supplementary normal approximation; not a definitive event-study test."
                )


# 5. Estimation-window robustness.
path = FILES["window_robustness"]

if path.exists():
    df = pd.read_csv(path)
    print(f"\nEstimation-window robustness columns: {list(df.columns)}")

    for _, row in df.iterrows():
        for col in df.columns:
            if col.lower() in {"event_date", "date", "ticker"}:
                continue

            value = row[col]
            if pd.isna(value):
                continue

            add_record(
                "estimation_window_robustness",
                str(row.get("event_date", row.get("date", "2024-06-04"))),
                str(row.get("ticker", "TCS.NS")),
                col,
                value,
                "Compare across estimation windows; p-values may be unadjusted."
            )


if not records:
    raise ValueError("No results were loaded. Check the input file paths.")

summary = pd.DataFrame(records)
summary.to_csv(OUTPUT, index=False)

print("\nRobustness summary saved:", OUTPUT)
print("Summary rows:", len(summary))
print("\nDiagnostic summary:")
print(summary.groupby("method").size().to_string())
print("\nTCS June 4 abnormal-return diagnostics:")
print(
    summary[
        (summary["ticker"] == "TCS.NS")
        & (summary["event_date"].astype(str).str.contains("2024-06-04"))
    ][["method", "metric", "value", "interpretation"]]
    .to_string(index=False)
)
