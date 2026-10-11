import numpy as np
import pandas as pd
from financial_events.event_study import market_model_event_study


def test_event_study_finds_positive_abnormal_return():
    rng = np.random.default_rng(7)
    n = 180
    dates = pd.bdate_range("2024-01-01", periods=n)
    market = rng.normal(.0003, .01, n)
    stock = .0002 + 1.2 * market + rng.normal(0, .002, n)
    event_idx = 150
    stock[event_idx] += .04
    result = market_model_event_study(
        pd.DataFrame({"date": dates, "return": stock}),
        pd.DataFrame({"date": dates, "return": market}),
        dates[event_idx],
    )
    assert result["estimation_observations"] >= 60
    assert result["car"] > .02
    assert len(result["event_returns"]) == 2
