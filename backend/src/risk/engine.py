"""
Kelly criterion + CVaR risk engine.
VIX veto threshold: 30 (crypto-adjusted, per Conflict SCH-003 resolution).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


class Verdict(str, Enum):
    APPROVE = "APPROVE"
    SCALE   = "SCALE"
    VETO    = "VETO"
    HALT    = "HALT"


@dataclass
class CopyTiers:
    auto:      float
    show_low:  float
    show_high: float


@dataclass
class RiskDecision:
    verdict:          Verdict
    scaled_notional:  float
    scale_factor:     float
    kelly_optimal:    float
    var_1d:           float
    cvar_1d:          float
    reasons:          list[str] = field(default_factory=list)
    copy_tiers:       Optional[CopyTiers] = None


_MAX_POS_PCT    = 0.20
_KELLY_CAP      = 0.25
_SPREAD_LIMIT   = 50.0
_DD_SCALE_TRIG  = 0.05
_DD_HALT_TRIG   = 0.10
_VIX_VETO       = 30.0


class PortfolioState:
    def __init__(self, nav: float):
        self._peak  = nav
        self._start = nav
        self._nav   = nav

    def update(self, delta: float) -> None:
        self._nav  += delta
        self._peak  = max(self._peak, self._nav)

    @property
    def drawdown(self) -> float:
        return (self._peak - self._nav) / (self._peak + 1e-9)

    def reset_daily(self) -> None:
        self._start = self._nav


class RiskEngine:
    def __init__(self, capital: float, portfolio: Optional[PortfolioState] = None):
        self._capital = capital
        self._port    = portfolio or PortfolioState(capital)

    def evaluate(
        self,
        notional:     float,
        signal_score: float,
        realized_vol: float,
        spread_bps:   float = 0.0,
        vix:          float = 20.0,
    ) -> RiskDecision:

        if self._port.drawdown >= _DD_HALT_TRIG:
            return RiskDecision(
                verdict=Verdict.HALT, scaled_notional=0.0, scale_factor=0.0,
                kelly_optimal=0.0, var_1d=0.0, cvar_1d=0.0,
                reasons=[f"Drawdown {self._port.drawdown:.1%} ≥ HALT {_DD_HALT_TRIG:.0%}"],
            )

        if spread_bps > _SPREAD_LIMIT:
            return RiskDecision(
                verdict=Verdict.VETO, scaled_notional=0.0, scale_factor=0.0,
                kelly_optimal=0.0, var_1d=0.0, cvar_1d=0.0,
                reasons=[f"Spread {spread_bps:.1f}bps > {_SPREAD_LIMIT}bps"],
            )

        if vix >= _VIX_VETO:
            return RiskDecision(
                verdict=Verdict.VETO, scaled_notional=0.0, scale_factor=0.0,
                kelly_optimal=0.0, var_1d=0.0, cvar_1d=0.0,
                reasons=[f"VIX {vix:.1f} ≥ {_VIX_VETO} (crypto threshold)"],
            )

        edge    = max(0.0, signal_score) * 0.10
        var_sq  = max(realized_vol**2, 1e-6)
        kelly_f = min(_KELLY_CAP, edge / var_sq)
        kelly_opt = kelly_f * self._capital

        dv      = realized_vol / np.sqrt(252)
        var_1d  = float(notional * dv * 1.645)
        cvar_1d = float(notional * dv * 2.326)

        scale   = 1.0
        reasons: list[str] = []
        max_n   = self._capital * _MAX_POS_PCT
        if notional > max_n:
            scale = max_n / notional
            reasons.append(f"Position capped at {_MAX_POS_PCT:.0%}")
        if self._port.drawdown >= _DD_SCALE_TRIG:
            dd_s  = 1.0 - self._port.drawdown / _DD_HALT_TRIG
            scale = min(scale, dd_s)
            reasons.append(f"Drawdown scale {dd_s:.0%}")

        tiers = CopyTiers(
            auto=self._capital * 0.01,
            show_low=self._capital * 0.05,
            show_high=self._capital * 0.25,
        )
        return RiskDecision(
            verdict=Verdict.SCALE if scale < 1.0 else Verdict.APPROVE,
            scaled_notional=notional * scale,
            scale_factor=scale,
            kelly_optimal=kelly_opt,
            var_1d=var_1d,
            cvar_1d=cvar_1d,
            reasons=reasons,
            copy_tiers=tiers,
        )
