import sys
sys.path.insert(0, '.')

import pandas as pd
from tests.test_clean_indicators import FakeFetcher
from src.indicators.clean_indicators import CleanIndicators
from src.indicators.get_indicators import IndicatorFetcher

print("=== TEST 1: CleanIndicators ===")
fetcher = FakeFetcher()
c = CleanIndicators(fetcher=fetcher)
out = c.get_clean_indicators(["pe", "sp"], "1y", ticker=None, freq="D")
print(f"Keys in output: {list(out.keys())}")
if "sp" in out:
    df_sp = out["sp"]
    print(f"SP columns: {df_sp.columns.tolist()}")
    print(f"SP shape: {df_sp.shape}")
    print(f"SP index tz: {df_sp.index.tz}")
    print(f"SP first 5 rows:\n{df_sp.head()}")
    print(f"SP last 5 rows:\n{df_sp.tail()}")

print("\n=== TEST 2: RSI ===")
idx = pd.DatetimeIndex([
    pd.Timestamp("2020-01-01", tz="UTC"),
    pd.Timestamp("2020-01-02", tz="UTC"),
    pd.Timestamp("2020-01-03", tz="UTC")
])
hist = pd.DataFrame({"Close": [1.0, 2.0, 1.5]}, index=idx)
print(f"Input hist:\n{hist}")
out = IndicatorFetcher._tech_rsi(hist, "rsi")
print(f"RSI output:\n{out}")
print(f"RSI empty: {out.empty}")