"""walk_forward.py — rolling walk-forward validation harness."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Callable
import numpy as np


@dataclass
class WFResult:
    window_idx: int
    train_bars: int
    test_bars:  int
    n_trades:   int
    gross_pnl:  float
    sharpe:     float
    sortino:    float
    max_dd:     float
    win_rate:   float
    avg_ret:    float
    elapsed_s:  float


@dataclass
class WFReport:
    results: list[WFResult] = field(default_factory=list)

    @property
    def mean_sharpe(self) -> float:
        vals = [r.sharpe for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_sortino(self) -> float:
        vals = [r.sortino for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def mean_max_dd(self) -> float:
        vals = [r.max_dd for r in self.results]
        return float(np.mean(vals)) if vals else 0.0

    @property
    def total_trades(self) -> int:
        return sum(r.n_trades for r in self.results)

    def summary(self) -> str:
        lines = [
            f"Walk-Forward Report — {len(self.results)} windows",
            f"  Mean Sharpe  : {self.mean_sharpe:+.3f}",
            f"  Mean Sortino : {self.mean_sortino:+.3f}",
            f"  Mean Max DD  : {self.mean_max_dd:.2%}",
            f"  Total trades : {self.total_trades}",
        ]
        for r in self.results:
            lines.append(
                f"  [{r.window_idx:02d}] "
                f"train={r.train_bars} test={r.test_bars} "
                f"trades={r.n_trades:3d} "
                f"pnl={r.gross_pnl:+.4f} "
                f"sharpe={r.sharpe:+.3f} "
                f"dd={r.max_dd:.2%} "
                f"({r.elapsed_s:.2f}s)"
            )
        return "\n".join(lines)


def _sharpe(rets: np.ndarray, ann: float = 252.0) -> float:
    if len(rets) < 2 or rets.std() < 1e-9:
        return 0.0
    return float(rets.mean() / rets.std() * np.sqrt(ann))


def _sortino(rets: np.ndarray, ann: float = 252.0) -> float:
    down = rets[rets < 0]
    dstd = down.std() if len(down) > 1 else 1e-9
    return float(rets.mean() / dstd * np.sqrt(ann)) if dstd > 0 else 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9)
    return float(-dd.min()) if len(dd) else 0.0


class WalkForwardBacktest:
    def __init__(
        self,
        train_bars: int = 200,
        test_bars:  int = 50,
        step_bars:  int = 50,
        cost_bps:   float = 6.0,
    ):
        self.train_bars = train_bars
        self.test_bars  = test_bars
        self.step_bars  = step_bars
        self.cost_bps   = cost_bps / 10_000.0

    def run(
        self,
        prices: np.ndarray,
        strategy_fn: Callable[[np.ndarray], float],
    ) -> WFReport:
        report = WFReport()
        n      = len(prices)
        start  = self.train_bars
        window = 0

        while start + self.test_bars <= n:
            t0       = time.time()
            train    = prices[:start]
            test_end = start + self.test_bars
            test_px  = prices[start:test_end]

            trade_rets: list[float] = []
            equity     = [1.0]
            prev_sig   = 0.0
            n_flips    = 0

            for i, px in enumerate(test_px):
                visible = np.concatenate([train, test_px[:i+1]])
                sig     = float(np.clip(strategy_fn(visible), -1.0, 1.0))
                if i > 0:
                    ret = (px / test_px[i-1] - 1.0) * prev_sig
                    if abs(sig - prev_sig) > 0.05:
                        ret     -= self.cost_bps
                        n_flips += 1
                    trade_rets.append(ret)
                    equity.append(equity[-1] * (1.0 + ret))
                prev_sig = sig

            r_arr = np.array(trade_rets) if trade_rets else np.zeros(1)
            report.results.append(WFResult(
                window_idx = window,
                train_bars = start,
                test_bars  = len(test_px),
                n_trades   = n_flips,
                gross_pnl  = float(r_arr.sum()),
                sharpe     = _sharpe(r_arr),
                sortino    = _sortino(r_arr),
                max_dd     = _max_drawdown(np.array(equity)),
                win_rate   = float((r_arr > 0).mean()),
                avg_ret    = float(r_arr.mean()),
                elapsed_s  = time.time() - t0,
            ))
            start  += self.step_bars
            window += 1

        return report
