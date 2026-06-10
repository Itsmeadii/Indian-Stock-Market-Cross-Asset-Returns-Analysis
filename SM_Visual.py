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
# GRAPH — ASSET GROWTH COMPARISON
# ─────────────────────────────────────────────────────

normalized = close / close.iloc[0] * 100

plt.figure()

for column in normalized.columns:
    plt.plot(normalized.index, normalized[column], label=column)

plt.title("Asset Growth Comparison (Base = 100)")
plt.xlabel("Date")
plt.ylabel("Growth Index")
plt.legend()
plt.grid()

plt.show()

