from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
import numpy as np

from forecasting.kalman import AdaptiveKalmanTracker, KalmanState

@dataclass
class Signal:
    symbol:       str
    price:        float
    score:        float       # composite [-1, +1]
    rsi:          float
    z_score:      float
    vol_regime:   str         # NORMAL / HIGH / EXTREME
    ob_imbalance: float
    kalman:       KalmanState
    components:   dict[str, float] = field(default_factory=dict)


class SignalEngine:
    """Momentum + mean-reversion + volatility + order-book signal engine."""

    MIN_BARS = 20
    WEIGHTS  = {"momentum": 0.35, "reversion": 0.35, "vol": 0.15, "ob": 0.15}

    def __init__(self, symbol: str):
        self.symbol  = symbol
        self._closes: deque[float] = deque(maxlen=256)
        self._kalman  = AdaptiveKalmanTracker()

    def update(self, price: float, bid: float = 0.0, ask: float = 0.0) -> Signal | None:
        self._closes.append(price)
        ks = self._kalman.update(price)
        if len(self._closes) < self.MIN_BARS:
            return None

        c   = np.array(self._closes)
        mom = self._momentum(c, ks)
        rev = self._reversion(c)
        vol = self._vol_signal(c)
        ob  = self._ob_signal(bid, ask)

        score = float(np.clip(
            self.WEIGHTS["momentum"] * mom +
            self.WEIGHTS["reversion"] * rev +
            self.WEIGHTS["vol"]       * vol +
            self.WEIGHTS["ob"]        * ob,
            -1.0, 1.0,
        ))
        return Signal(
            symbol=self.symbol, price=price, score=score,
            rsi=self._rsi(c), z_score=self._z(c),
            vol_regime=self._vol_regime(c),
            ob_imbalance=ob, kalman=ks,
            components={"momentum": mom, "reversion": rev, "vol": vol, "ob": ob},
        )

    @staticmethod
    def _rsi(c: np.ndarray, p: int = 14) -> float:
        d  = np.diff(c[-p-1:])
        ag = np.maximum(d, 0.0).mean()
        al = np.maximum(-d, 0.0).mean()
        return 100.0 - 100.0 / (1.0 + ag / (al + 1e-9))

    @staticmethod
    def _z(c: np.ndarray, w: int = 20) -> float:
        seg = c[-w:]
        return float((seg[-1] - seg.mean()) / (seg.std() + 1e-9))

    @staticmethod
    def _vol_regime(c: np.ndarray) -> str:
        r = np.diff(np.log(c[-22:]))
        v = r.std() * np.sqrt(252 * 288)
        if v > 2.0:  return "EXTREME"
        if v > 0.8:  return "HIGH"
        return "NORMAL"

    def _momentum(self, c: np.ndarray, ks: KalmanState) -> float:
        rsi_n = (self._rsi(c) - 50.0) / 50.0
        z     = self._z(c)
        vel_n = float(np.tanh(ks.velocity / (c[-1] * 0.001 + 1e-9)))
        return float(np.clip(0.5 * rsi_n + 0.3 * z + 0.2 * vel_n, -1.0, 1.0))

    def _reversion(self, c: np.ndarray) -> float:
        return float(np.clip(-np.tanh(self._z(c) * 0.8), -1.0, 1.0))

    def _vol_signal(self, c: np.ndarray) -> float:
        return {"NORMAL": 0.0, "HIGH": -0.15, "EXTREME": -0.40}[self._vol_regime(c)]

    @staticmethod
    def _ob_signal(bid: float, ask: float) -> float:
        if bid == 0 or ask == 0:
            return 0.0
        mid   = (bid + ask) / 2.0
        spbps = (ask - bid) / mid * 10_000
        return float(np.clip(-np.tanh(spbps / 10.0 - 1.0), -1.0, 1.0))
