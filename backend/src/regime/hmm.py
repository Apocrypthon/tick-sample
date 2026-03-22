from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from collections import deque
import numpy as np


class Regime(str, Enum):
    BULL     = "BULL"
    BEAR     = "BEAR"
    VOLATILE = "VOLATILE"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class RegimeState:
    regime: Regime
    probabilities: dict[Regime, float]
    position_multiplier: float


_MULTIPLIERS = {
    Regime.BULL:     1.00,
    Regime.SIDEWAYS: 0.60,
    Regime.VOLATILE: 0.35,
    Regime.BEAR:     0.10,
}


class HMMRegimeDetector:
    MIN_TRAIN = 60
    N_STATES  = 4

    def __init__(self):
        from hmmlearn import hmm as hmmlib
        self._model = hmmlib.GaussianHMM(
            n_components=self.N_STATES, covariance_type="diag",
            n_iter=100, random_state=42,
        )
        self._fitted  = False
        self._returns: deque[float] = deque(maxlen=512)
        self._state_map: dict[int, Regime] = {}

    def update(self, price: float) -> RegimeState | None:
        if self._returns:
            self._returns.append(price / list(self._returns)[-1] - 1.0)
        else:
            self._returns.append(0.0)
            return None

        if len(self._returns) < self.MIN_TRAIN:
            return None
        if len(self._returns) % 20 == 0 or not self._fitted:
            self._fit()
        return self._predict() if self._fitted else None

    def _fit(self) -> None:
        rets = np.array(self._returns)
        X    = np.column_stack([rets, np.abs(rets)])
        try:
            self._model.fit(X)
            self._fitted    = True
            means           = self._model.means_[:, 0]
            vol_means       = self._model.means_[:, 1]
            order           = np.argsort(means)
            high_vol_idx    = int(np.argmax(vol_means))
            mapped: dict[int, Regime] = {}
            for rank, idx in enumerate(order):
                if int(idx) == high_vol_idx:
                    mapped[int(idx)] = Regime.VOLATILE
                elif rank == 0:
                    mapped[int(idx)] = Regime.BEAR
                elif rank == len(order) - 1:
                    mapped[int(idx)] = Regime.BULL
                else:
                    mapped[int(idx)] = Regime.SIDEWAYS
            self._state_map = mapped
        except Exception:
            self._fitted = False

    def _predict(self) -> RegimeState:
        rets = np.array(self._returns)
        X    = np.column_stack([rets, np.abs(rets)])
        try:
            probs     = self._model.predict_proba(X)[-1]
            state_idx = int(np.argmax(probs))
            regime    = self._state_map.get(state_idx, Regime.SIDEWAYS)
            prob_map  = {
                self._state_map.get(i, Regime.SIDEWAYS): float(probs[i])
                for i in range(self.N_STATES)
            }
        except Exception:
            regime   = Regime.SIDEWAYS
            prob_map = {r: 0.25 for r in Regime}
        return RegimeState(regime, prob_map, _MULTIPLIERS[regime])
