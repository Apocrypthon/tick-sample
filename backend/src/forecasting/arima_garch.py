"""Fixed ARIMA(1,1,1) + EWMA-GARCH vol. Conflict FND-B: Option A — fixed order."""
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class Forecast:
    mean: float
    lower: float
    upper: float
    vol_daily: float


class ARIMAForecaster:
    MIN_BARS = 30

    def __init__(self, horizon: int = 5, ewma_span: int = 20):
        self._horizon = horizon
        self._span    = ewma_span
        self._returns: deque[float] = deque(maxlen=256)
        self._prices:  deque[float] = deque(maxlen=256)

    def update(self, price: float) -> Forecast | None:
        if self._prices:
            self._returns.append(price / self._prices[-1] - 1.0)
        self._prices.append(price)
        if len(self._prices) < self.MIN_BARS:
            return None
        return self._fit()

    def _fit(self) -> Forecast:
        vol = self._ewma_vol()
        try:
            from statsmodels.tsa.arima.model import ARIMA
            arr = np.array(self._prices)
            res = ARIMA(arr, order=(1, 1, 1)).fit(
                method_kwargs={"warn_convergence": False}
            )
            fc  = res.get_forecast(steps=self._horizon)
            m   = float(fc.predicted_mean.iloc[-1])
            ci  = fc.conf_int(alpha=0.05)
            lo, hi = float(ci.iloc[-1, 0]), float(ci.iloc[-1, 1])
        except Exception:
            m  = float(self._prices[-1]) * (1.0 + np.mean(self._returns) * self._horizon)
            lo = m * 0.97
            hi = m * 1.03
        return Forecast(mean=m, lower=lo, upper=hi, vol_daily=vol)

    def _ewma_vol(self) -> float:
        r     = np.array(list(self._returns)[-self._span:])
        alpha = 2.0 / (self._span + 1.0)
        var   = float(np.var(r)) if len(r) > 1 else 0.0004
        for v in r:
            var = alpha * v**2 + (1.0 - alpha) * var
        return float(np.sqrt(max(var, 1e-10) * 252))
