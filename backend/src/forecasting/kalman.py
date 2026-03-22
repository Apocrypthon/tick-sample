from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np


@dataclass
class KalmanState:
    level: float
    velocity: float
    variance: float
    innovation: float
    forecast_5: float


class AdaptiveKalmanTracker:
    """Noise-filtered price tracker with velocity + 5-bar forecast."""

    # F: state transition matrix — level += velocity each step
    _F = np.array([[1.0, 1.0], [0.0, 1.0]])
    _H = np.array([[1.0, 0.0]])

    def __init__(self, process_noise: float = 1e-3, obs_noise: float = 1e-2):
        self._q = process_noise
        self._r = obs_noise
        self._x = np.zeros(2)
        self._p = np.eye(2)
        self._initialized = False

    def update(self, price: float) -> KalmanState:
        if not self._initialized:
            self._x = np.array([price, 0.0])
            self._initialized = True
            return KalmanState(price, 0.0, 1.0, 0.0, price)

        x_pred = self._F @ self._x
        p_pred = self._F @ self._p @ self._F.T + self._q * np.eye(2)

        innov = price - (self._H @ x_pred)[0]
        s     = float((self._H @ p_pred @ self._H.T)[0, 0] + self._r)
        k     = (p_pred @ self._H.T / s).flatten()

        self._x = x_pred + k * innov
        self._p = (np.eye(2) - np.outer(k, self._H)) @ p_pred
        # Adaptive: widen process noise when innovation is large
        self._q = float(np.clip(abs(innov) / (price + 1e-9) * 0.5, 1e-5, 0.1))

        return KalmanState(
            level=float(self._x[0]),
            velocity=float(self._x[1]),
            variance=float(self._p[0, 0]),
            innovation=float(innov),
            forecast_5=float(self._x[0] + 5.0 * self._x[1]),
        )

    @property
    def is_ready(self) -> bool:
        return self._initialized
