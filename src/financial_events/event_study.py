import numpy as np
import pandas as pd
import statsmodels.api as sm


def market_model_event_study(
    stock_returns: pd.DataFrame,
    market_returns: pd.DataFrame,
    event_date,
    estimation_window=(-120, -21),
    event_window=(0, 1),
    min_estimation_observations=60,
) -> dict:
    """Baseline market-model event study for one event.

    Inputs each contain date and return columns, one row per trading session.
    event_date must already be mapped to a valid aligned trading session.
    Overlapping events, confounders, clustered inference, and multiple testing
    are intentionally outside this first baseline.
    """
    for name, frame in (("stock_returns", stock_returns), ("market_returns", market_returns)):
        if not {"date", "return"}.issubset(frame.columns):
            raise ValueError(f"{name} must contain date and return columns")

    stock = stock_returns[["date", "return"]].copy()
    market = market_returns[["date", "return"]].copy()
    stock["date"] = pd.to_datetime(stock["date"]).dt.normalize()
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()
    stock = stock.rename(columns={"return": "stock_return"})
    market = market.rename(columns={"return": "market_return"})
    data = stock.merge(market, on="date", how="inner").sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError("Expected one return per date in each input")

    event_date = pd.Timestamp(event_date).normalize()
    positions = data.index[data["date"] == event_date].tolist()
    if not positions:
        raise ValueError("event_date is not present in aligned trading-session data")
    data["event_time"] = np.arange(len(data)) - positions[0]

    est_lo, est_hi = estimation_window
    evt_lo, evt_hi = event_window
    est = data[data["event_time"].between(est_lo, est_hi)].dropna(
        subset=["stock_return", "market_return"]
    )
    evt = data[data["event_time"].between(evt_lo, evt_hi)].dropna(
        subset=["stock_return", "market_return"]
    )
    if len(est) < min_estimation_observations:
        raise ValueError(f"Insufficient estimation observations: {len(est)}")
    if set(evt["event_time"]) != set(range(evt_lo, evt_hi + 1)):
        raise ValueError("Event window has missing trading-session observations")

    model = sm.OLS(est["stock_return"], sm.add_constant(est["market_return"])).fit()
    evt = evt.copy()
    evt["expected_return"] = model.predict(sm.add_constant(evt["market_return"], has_constant="add"))
    evt["abnormal_return"] = evt["stock_return"] - evt["expected_return"]
    return {
        "event_date": event_date.date().isoformat(),
        "estimation_window": [est_lo, est_hi],
        "event_window": [evt_lo, evt_hi],
        "estimation_observations": int(len(est)),
        "alpha": float(model.params["const"]),
        "beta": float(model.params["market_return"]),
        "car": float(evt["abnormal_return"].sum()),
        "event_returns": evt[["date", "event_time", "stock_return", "market_return",
                              "expected_return", "abnormal_return"]]
            .assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d")).to_dict("records"),
    }
