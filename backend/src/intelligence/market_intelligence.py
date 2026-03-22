from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional
import numpy as np


@dataclass
class MarketSnapshot:
    regime:       str
    fear_greed:   int
    vix_proxy:    float
    avg_vol_1d:   float
    btc_eth_corr: float
    btc_sol_corr: float
    vol_structure: dict[str, float]
    narrative:    str


class MarketIntelligence:
    MIN_BARS = 50

    def __init__(self, symbols: list[str] | None = None):
        self._syms    = symbols or ["BTC-USD", "ETH-USD", "SOL-USD"]
        self._prices  = {s: deque(maxlen=256) for s in self._syms}
        self._returns = {s: deque(maxlen=256) for s in self._syms}
        self._vix     = 20.0

    def update(self, prices: dict[str, float], vix: float = 20.0) -> Optional[MarketSnapshot]:
        self._vix = vix
        for sym, px in prices.items():
            if sym not in self._prices:
                self._prices[sym]  = deque(maxlen=256)
                self._returns[sym] = deque(maxlen=256)
            if self._prices[sym]:
                self._returns[sym].append(px / self._prices[sym][-1] - 1.0)
            self._prices[sym].append(px)

        if len(self._returns.get("BTC-USD", [])) < self.MIN_BARS:
            return None

        btc_rets   = self._returns["BTC-USD"]
        regime     = self._regime(btc_rets)
        fg         = self._fear_greed(vix, btc_rets)
        vol_struct = {s: self._ewma_vol(self._returns[s])
                      for s in self._syms if len(self._returns.get(s, [])) >= 20}
        avg_vol    = float(np.mean(list(vol_struct.values()))) if vol_struct else 0.0

        return MarketSnapshot(
            regime=regime,
            fear_greed=fg,
            vix_proxy=vix,
            avg_vol_1d=avg_vol,
            btc_eth_corr=self._corr("BTC-USD", "ETH-USD"),
            btc_sol_corr=self._corr("BTC-USD", "SOL-USD"),
            vol_structure=vol_struct,
            narrative=self._narrative(regime, fg, vix, avg_vol),
        )

    def _regime(self, rets: deque) -> str:
        r = np.array(rets)[-50:]
        try:
            from hmmlearn import hmm as hmmlib
            m = hmmlib.GaussianHMM(n_components=2, covariance_type="diag",
                                   n_iter=20, random_state=0)
            m.fit(r.reshape(-1, 1))
            bear = int(np.argmin(m.means_[:, 0]))
            return "RISK-OFF" if m.predict(r.reshape(-1, 1))[-1] == bear else "RISK-ON"
        except Exception:
            return "RISK-OFF" if (r < -0.005).mean() > 0.55 else "RISK-ON"

    @staticmethod
    def _fear_greed(vix: float, rets: deque) -> int:
        vc  = max(0.0, min(1.0, (50.0 - vix) / 35.0))
        r14 = np.array(rets)[-14:]
        mc  = max(0.0, min(1.0, r14.mean() / 0.01 + 0.5))
        return int(np.clip((0.6 * vc + 0.4 * mc) * 100, 0, 100))

    def _corr(self, a: str, b: str, w: int = 60) -> float:
        ra = list(self._returns.get(a, []))[-w:]
        rb = list(self._returns.get(b, []))[-w:]
        n  = min(len(ra), len(rb))
        return float(np.corrcoef(ra[-n:], rb[-n:])[0, 1]) if n >= 10 else 0.0

    @staticmethod
    def _ewma_vol(rets: deque, span: int = 20) -> float:
        r     = np.array(list(rets)[-span:])
        alpha = 2.0 / (span + 1)
        var   = float(np.var(r)) if len(r) > 1 else 4e-4
        for v in r:
            var = alpha * v**2 + (1.0 - alpha) * var
        return float(np.sqrt(max(var, 1e-10) * 252))

    @staticmethod
    def _narrative(regime: str, fg: int, vix: float, vol: float) -> str:
        lbl = ("EXTREME FEAR 😱" if fg < 20 else "FEAR 😟" if fg < 40 else
               "NEUTRAL 😐" if fg < 60 else "GREED 😄" if fg < 80 else "EXTREME GREED 🤑")
        return (f"Regime: {regime}  F/G: {fg}/100 — {lbl}  "
                f"VIX proxy: ${vix:.1f}  Avg 1d vol: {vol:.0%}")
