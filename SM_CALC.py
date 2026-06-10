import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────

tickers = {
    "Nifty50":         "^NSEI",
    "NiftySmallcap":   "^CNXSC",
    "Nasdaq":          "^IXIC",
    "Gold":            "GC=F",
    "ICICI_Prudential": "0P0000XVU1.BO"
}

YEARS        = 5
RISK_FREE    = 6.0
TRADING_DAYS = 252

# ─────────────────────────────────────────────────────
# SECTION 1 — FETCH DATA
# ─────────────────────────────────────────────────────

print("Fetching 5-year data...\n")

raw = yf.download(
    list(tickers.values()),
    period="5y",
    auto_adjust=True,
    progress=False,
    threads=False   
)

# Handle multi-index safely
if isinstance(raw.columns, pd.MultiIndex):
    close = raw["Close"].copy()
else:
    close = raw.copy()

close.columns = list(tickers.keys())

# Drop rows where ALL values missing
close.dropna(how="all", inplace=True)

close = close.ffill()

valid_counts = close.notna().sum()
close = close.loc[:, valid_counts > len(close) * 0.2]

# Final clean
close = close.dropna()
close = close.sort_index()

print(f"Date Range  : {close.index[0].date()} → {close.index[-1].date()}")
print(f"Trading Days: {len(close)}\n")
print(f"Assets loaded: {list(close.columns)}\n")

# ─────────────────────────────────────────────────────
# SECTION 2 — CALCULATIONS
# ─────────────────────────────────────────────────────

abs_return    = (close.iloc[-1] / close.iloc[0] - 1) * 100
cagr          = ((close.iloc[-1] / close.iloc[0]) ** (1 / YEARS) - 1) * 100
daily_returns = close.pct_change().dropna()
volatility    = daily_returns.std() * np.sqrt(TRADING_DAYS) * 100
sharpe = ((cagr/100 - RISK_FREE/100) / (volatility/100))

# ─────────────────────────────────────────────────────
# SECTION 3 — SUMMARY
# ─────────────────────────────────────────────────────

W = 72

def divider():
    print("=" * W)

def header(title):
    divider()
    print(title.center(W))
    divider()

header("5-YEAR PERFORMANCE SUMMARY")

summary = pd.DataFrame({
    "Abs Return (%)": abs_return.round(2),
    "CAGR (%)":       cagr.round(2),
    "Volatility (%)": volatility.round(2),
    "Sharpe Ratio":   sharpe.round(2),
})

print(summary.to_string())
divider()

# ─────────────────────────────────────────────────────
# FUTURE RETURN ESTIMATOR
# ─────────────────────────────────────────────────────

def risk_label(vol):
    if vol < 15:   return "Non-Risky"
    elif vol < 25: return "Neutral"
    else:          return "Risky"

print("\n" + "=" * W)
print("FUTURE RETURN ESTIMATOR".center(W))
print("=" * W)

amount = float(input("  Enter investment amount (₹): "))
years  = float(input("  Enter investment period (in years): "))

print()
divider()
print(f"  Invested : ₹{amount:,.0f}   |   Period : {years} years")
print("-" * W)
print(f"  {'Asset':<18} {'CAGR':>8}  {'Est. Value':>16}  {'Total Gain':>14}  {'Risk':>10}")
print("-" * W)

results = {}
for asset in close.columns:
    rate = cagr[asset] / 100
    future_val = amount * ((1 + rate) ** years)
    results[asset] = future_val

for asset in sorted(results, key=results.get, reverse=True):
    gain = results[asset] - amount
    print(f"  {asset:<18} {cagr[asset]:>7.1f}%  "
          f"₹{results[asset]:>14,.0f}  "
          f"₹{gain:>12,.0f}  "
          f"{risk_label(volatility[asset]):>10}")

divider()