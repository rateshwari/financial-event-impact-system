
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
import statsmodels.api as sm

SOURCE = Path("data/processed/daily_returns.csv")
OUTPUT = Path("data/processed/tcs_hc3_inference.csv")

STOCK = "TCS.NS"
MARKET = "^NSEI"
EVENT_DATE = pd.Timestamp("2024-06-04")

# Original market-model estimation window: trading days -120 to -21.
LOOKBACK = 120
GAP = 20

df = pd.read_csv(SOURCE, parse_dates=["date"])
df["date"] = df["date"].dt.normalize()

wide = (
    df.pivot(index="date", columns="ticker", values="daily_return")
    .sort_index()
)

dates = list(wide.index)
date_to_idx = {date: i for i, date in enumerate(dates)}

if EVENT_DATE not in date_to_idx:
    raise ValueError(f"{EVENT_DATE.date()} is not in the trading-date index.")

event_idx = date_to_idx[EVENT_DATE]
start_idx = event_idx - LOOKBACK
end_idx = event_idx - GAP

if start_idx < 0:
    raise ValueError("Insufficient history for the estimation window.")

estimation_dates = dates[start_idx:end_idx]
est = wide.loc[estimation_dates, [STOCK, MARKET]].dropna()

if len(est) < 80:
    raise ValueError(
        f"Too few estimation observations: {len(est)}"
    )

# Fit market model with HC3 heteroskedasticity-consistent covariance.
X = sm.add_constant(est[MARKET], has_constant="add")
y = est[STOCK]

ols = sm.OLS(y, X)
classical_fit = ols.fit()
hc3_fit = ols.fit(cov_type="HC3")

# Event-day actual, expected, and abnormal returns.
event = wide.loc[EVENT_DATE, [STOCK, MARKET]]

actual = float(event[STOCK])
market_return = float(event[MARKET])

x_event = np.array([1.0, market_return])
alpha = float(classical_fit.params["const"])
beta = float(classical_fit.params[MARKET])

expected = alpha + beta * market_return
abnormal = actual - expected

# Robust variance of the estimated conditional mean at the event-day
# market return, using the HC3 covariance matrix.
cov_hc3 = np.asarray(hc3_fit.cov_params())
mean_variance_hc3 = float(x_event @ cov_hc3 @ x_event)

# Supplementary prediction-error variance:
# historical residual variance + HC3 variance of the estimated mean.
# This is a normal-approximation diagnostic, not an exact finite-sample
# event-study test.
residual_variance = float(np.mean(classical_fit.resid ** 2))
prediction_variance = residual_variance + mean_variance_hc3

if prediction_variance <= 0:
    raise ValueError("Prediction variance must be positive.")

robust_prediction_se = np.sqrt(prediction_variance)
z_stat = abnormal / robust_prediction_se
p_value = 2 * norm.sf(abs(z_stat))

result = pd.DataFrame([{
    "event_date": EVENT_DATE.date().isoformat(),
    "ticker": STOCK,
    "estimation_observations": len(est),
    "actual_return": actual,
    "market_return": market_return,
    "alpha": alpha,
    "beta": beta,
    "expected_return": expected,
    "abnormal_return": abnormal,
    "residual_variance": residual_variance,
    "hc3_mean_prediction_variance": mean_variance_hc3,
    "robust_prediction_standard_error": robust_prediction_se,
    "normal_approx_z_statistic": z_stat,
    "normal_approx_two_sided_p": p_value,
}])

result.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print("\nTCS HC3 ROBUSTNESS DIAGNOSTIC")
print(f"Event date: {EVENT_DATE.date()}")
print(f"Estimation observations: {len(est)}")
print(f"Actual return: {actual:.4%}")
print(f"NIFTY return: {market_return:.4%}")
print(f"Expected return: {expected:.4%}")
print(f"Abnormal return: {abnormal:.4%}")
print(f"HC3 robust prediction SE: {robust_prediction_se:.4%}")
print(f"Normal-approximation z statistic: {z_stat:.4f}")
print(f"Normal-approximation two-sided p-value: {p_value:.6f}")

print(
    "\nInterpretation: this combines HC3 uncertainty in the fitted "
    "conditional mean with historical residual variance. The resulting "
    "normal-approximation p-value is a supplementary diagnostic, not "
    "a definitive event-study significance test. It does not account "
    "for serial dependence, event selection, or multiple testing."
)
