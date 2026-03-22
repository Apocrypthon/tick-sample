"""Alpha factor library — pure numpy fallback (pandas-ta optional)."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


class AlphaFactors:

    @staticmethod
    def compute(
        close:  np.ndarray,
        high:   np.ndarray | None = None,
        low:    np.ndarray | None = None,
        volume: np.ndarray | None = None,
    ) -> pd.DataFrame:
        n = len(close)
        df = pd.DataFrame({
            "close":  close,
            "high":   high   if high   is not None else close,
            "low":    low    if low    is not None else close,
            "volume": volume if volume is not None else np.ones(n),
        })

        # ── optional pandas-ta block ──────────────────────────────────────
        try:
            import pandas_ta as ta  # noqa: F401
            df.ta.rsi(length=14,   append=True)
            df.ta.macd(            append=True)
            df.ta.bbands(length=20, append=True)
            df.ta.atr(length=14,   append=True)
            df.ta.obv(             append=True)
            df.ta.stoch(           append=True)
            df.ta.cci(length=20,   append=True)
            df.ta.willr(length=14, append=True)
            df.ta.mfi(length=14,   append=True)
        except (ImportError, Exception):
            pass  # pure-numpy factors below are sufficient

        # ── pure-numpy factors (always computed) ──────────────────────────
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            log_ret = np.log(np.maximum(close, 1e-9) / np.maximum(np.roll(close, 1), 1e-9))
            log_ret[0] = 0.0

            ser = pd.Series(close)
            df["alpha_log_ret"]   = log_ret
            df["alpha_ret_5"]     = ser.pct_change(5).values
            df["alpha_ret_20"]    = ser.pct_change(20).values
            df["alpha_vol_5"]     = (
                pd.Series(log_ret).rolling(5).std().fillna(0.0).values * np.sqrt(252)
            )
            df["alpha_vol_20"]    = (
                pd.Series(log_ret).rolling(20).std().fillna(0.0).values * np.sqrt(252)
            )
            ma5  = ser.rolling(5).mean()
            ma20 = ser.rolling(20).mean()
            std5  = ser.rolling(5).std().replace(0.0, 1e-9)
            std20 = ser.rolling(20).std().replace(0.0, 1e-9)
            df["alpha_z_5"]       = ((ser - ma5)  / std5).values
            df["alpha_z_20"]      = ((ser - ma20) / std20).values
            df["alpha_amihud"]    = (
                np.abs(log_ret) / (df["volume"].values * np.maximum(close, 1e-9))
            )
            df["alpha_vol_ratio"] = (
                df["alpha_vol_5"] / np.maximum(df["alpha_vol_20"], 1e-9)
            )

            # ── RSI (pure numpy, no TA-Lib) ──────────────────────────────
            rsi_vals = np.full(n, 50.0)
            if n >= 15:
                d  = np.diff(close)
                for i in range(14, n):
                    window = d[max(0, i-14):i]
                    ag = window[window > 0].mean() if (window > 0).any() else 1e-9
                    al = (-window[window < 0]).mean() if (window < 0).any() else 1e-9
                    rsi_vals[i] = 100.0 - 100.0 / (1.0 + ag / al)
            df["alpha_rsi"] = rsi_vals

        df.iloc[0] = np.nan
        return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def latest_normalized(df: pd.DataFrame) -> dict[str, float]:
        row  = df.iloc[-1]
        keys = ["alpha_z_5", "alpha_z_20", "alpha_vol_ratio", "alpha_amihud", "alpha_rsi"]
        return {k: float(np.tanh(row.get(k, 0.0))) for k in keys}
