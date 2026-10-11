import pandas as pd
from financial_events.returns import calculate_simple_returns


def test_returns_calculated_per_ticker():
    prices = pd.DataFrame({
        "date":["2025-01-01","2025-01-02","2025-01-01","2025-01-02"],
        "ticker":["AAA","AAA","BBB","BBB"],
        "close":[100,110,50,45],
    })
    result = calculate_simple_returns(prices)
    aaa = result[result.ticker == "AAA"].sort_values("date")
    bbb = result[result.ticker == "BBB"].sort_values("date")
    assert pd.isna(aaa.iloc[0]["return"])
    assert abs(aaa.iloc[1]["return"] - .10) < 1e-9
    assert abs(bbb.iloc[1]["return"] + .10) < 1e-9
