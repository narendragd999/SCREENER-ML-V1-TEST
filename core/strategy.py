"""Causal daily indicators shared by scanning and historical evaluation."""
import numpy as np
import pandas as pd


def indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    bars = df.copy()
    close = bars["Close"]
    bars["ema9"] = close.ewm(span=9, adjust=False).mean()
    bars["sma50"] = close.rolling(50, min_periods=50).mean()
    bars["sma200"] = close.rolling(200, min_periods=200).mean()
    prev = close.shift()
    tr = pd.concat([bars.High - bars.Low, (bars.High - prev).abs(),
                    (bars.Low - prev).abs()], axis=1).max(axis=1)
    bars["atr14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    bars["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    bars.loc[(loss == 0) & (gain > 0), "rsi14"] = 100
    bars.loc[(loss == 0) & (gain == 0), "rsi14"] = 50
    # A signal's volume is compared with PREVIOUS sessions, not itself.
    bars["volume_ratio"] = bars.Volume / bars.Volume.shift().rolling(20).mean().replace(0, np.nan)
    bars["turnover20"] = (close * bars.Volume).rolling(20).mean()
    window = 60
    x = np.arange(window, dtype=float)
    centered = x - x.mean()
    denom = float(np.dot(centered, centered))
    bars["lr_slope_pct"] = close.rolling(window).apply(
        lambda y: float(np.dot(centered, y) / denom / y.mean() * 100), raw=True)
    return bars


def confirmed_breakout_mask(df: pd.DataFrame) -> pd.Series:
    """True at the confirmed candle; never uses a future candle's values."""
    bars = df if "ema9" in df else indicator_frame(df)
    breakout = (bars.Close.shift() <= bars.ema9.shift()) & (bars.Close > bars.ema9)
    confirmed = (breakout.shift(fill_value=False) & (bars.Close > bars.Close.shift())
                 & (bars.High > bars.High.shift()) & (bars.Close > bars.ema9))
    confirmed.iloc[:30] = False
    return confirmed.fillna(False)


def trend_regime_mask(df: pd.DataFrame) -> pd.Series:
    bars = df if "lr_slope_pct" in df and "sma200" in df else indicator_frame(df)
    regimes = pd.Series("UNKNOWN", index=bars.index, dtype=object)
    known = bars.sma200.notna() & bars.lr_slope_pct.notna()
    regimes.loc[known] = "SIDEWAYS"
    up = (known & (bars.Close > bars.sma50) & (bars.Close > bars.sma200)
          & (bars.sma50 > bars.sma200) & (bars.lr_slope_pct > 0)
          & (bars.sma50 > bars.sma50.shift(5)))
    down = (known & (bars.Close < bars.sma50) & (bars.Close < bars.sma200)
            & (bars.sma50 < bars.sma200) & (bars.lr_slope_pct < 0))
    regimes.loc[up] = "UPTREND"
    regimes.loc[down] = "DOWNTREND"
    return regimes


def compute_trend(df: pd.DataFrame) -> dict:
    bars = indicator_frame(df)
    row = bars.iloc[-1]
    def number(key):
        return round(float(row[key]), 4) if pd.notna(row[key]) else None
    return {"trend_regime": str(trend_regime_mask(bars).iloc[-1]),
            "sma50": number("sma50"), "sma200": number("sma200"),
            "lr_slope_pct": number("lr_slope_pct"),
            "above_50sma": bool(row.Close > row.sma50) if pd.notna(row.sma50) else None,
            "above_200sma": bool(row.Close > row.sma200) if pd.notna(row.sma200) else None,
            "trend_history_sufficient": bool(pd.notna(row.sma200))}
