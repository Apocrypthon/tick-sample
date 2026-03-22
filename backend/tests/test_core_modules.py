"""Unit tests — all new Phase 1-5 modules. Run: uv run pytest tests/ -v"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import numpy as np


# ── Kalman ────────────────────────────────────────────────────────────────────

class TestKalmanTracker:
    def test_first_update_returns_state(self):
        from forecasting.kalman import AdaptiveKalmanTracker
        kf = AdaptiveKalmanTracker()
        s  = kf.update(65000.0)
        assert s.level == 65000.0
        assert s.velocity == 0.0
        assert s.forecast_5 == 65000.0

    def test_level_tracks_price_closely(self):
        from forecasting.kalman import AdaptiveKalmanTracker
        kf = AdaptiveKalmanTracker()
        prices = 60000 + np.cumsum(np.random.randn(80) * 50)
        for p in prices:
            s = kf.update(float(p))
        assert abs(s.level - prices[-1]) < prices[-1] * 0.05

    def test_forecast_5_is_level_plus_5x_velocity(self):
        from forecasting.kalman import AdaptiveKalmanTracker
        kf = AdaptiveKalmanTracker()
        for p in np.linspace(60000, 65000, 50):
            s = kf.update(float(p))
        assert s.velocity > 0
        assert abs(s.forecast_5 - (s.level + 5 * s.velocity)) < 1e-6

    def test_innovation_is_nonzero_after_price_jump(self):
        from forecasting.kalman import AdaptiveKalmanTracker
        kf = AdaptiveKalmanTracker()
        for _ in range(20):
            kf.update(60000.0)
        s = kf.update(70000.0)
        assert abs(s.innovation) > 100


# ── SignalEngine ──────────────────────────────────────────────────────────────

class TestSignalEngine:
    def _warm(self, n=40):
        from signals.signal_engine import SignalEngine
        eng = SignalEngine("BTC-USD")
        sig = None
        for p in 60000 + np.cumsum(np.random.randn(n) * 200):
            sig = eng.update(float(p))
        return sig

    def test_returns_none_before_min_bars(self):
        from signals.signal_engine import SignalEngine
        eng = SignalEngine("BTC-USD")
        for p in [60000.0] * 5:
            assert eng.update(p) is None

    def test_score_within_unit_interval(self):
        sig = self._warm()
        assert sig is not None
        assert -1.0 <= sig.score <= 1.0

    def test_rsi_within_0_100(self):
        sig = self._warm()
        assert 0.0 <= sig.rsi <= 100.0

    def test_vol_regime_is_valid_label(self):
        sig = self._warm()
        assert sig.vol_regime in ("NORMAL", "HIGH", "EXTREME")

    def test_components_sum_approximates_score(self):
        from signals.signal_engine import SignalEngine
        eng = SignalEngine("BTC-USD")
        sig = None
        for p in 60000 + np.cumsum(np.random.randn(40) * 200):
            sig = eng.update(float(p))
        w = eng.WEIGHTS
        manual = (
            w["momentum"] * sig.components["momentum"] +
            w["reversion"] * sig.components["reversion"] +
            w["vol"]       * sig.components["vol"] +
            w["ob"]        * sig.components["ob"]
        )
        import numpy as np2
        assert abs(np.clip(manual, -1.0, 1.0) - sig.score) < 1e-9


# ── RiskEngine ────────────────────────────────────────────────────────────────

class TestRiskEngine:
    def test_approve_on_positive_signal_normal_vix(self):
        from risk.engine import RiskEngine, Verdict
        dec = RiskEngine(10_000.0).evaluate(
            notional=300, signal_score=0.5, realized_vol=0.6, vix=20.0
        )
        assert dec.verdict in (Verdict.APPROVE, Verdict.SCALE)

    def test_veto_on_vix_above_30(self):
        from risk.engine import RiskEngine, Verdict
        dec = RiskEngine(10_000.0).evaluate(
            notional=300, signal_score=0.5, realized_vol=0.6, vix=31.0
        )
        assert dec.verdict == Verdict.VETO

    def test_veto_on_wide_spread(self):
        from risk.engine import RiskEngine, Verdict
        dec = RiskEngine(10_000.0).evaluate(
            notional=300, signal_score=0.5, realized_vol=0.6,
            vix=20.0, spread_bps=60.0
        )
        assert dec.verdict == Verdict.VETO

    def test_halt_on_10pct_drawdown(self):
        from risk.engine import RiskEngine, Verdict, PortfolioState
        port = PortfolioState(nav=10_000.0)
        port.update(-1_100.0)  # 11% drawdown — clearly above 10% halt threshold
        dec = RiskEngine(10_000.0, portfolio=port).evaluate(
            notional=300, signal_score=0.5, realized_vol=0.6, vix=20.0
        )
        assert dec.verdict == Verdict.HALT

    def test_scale_on_oversized_position(self):
        from risk.engine import RiskEngine, Verdict
        dec = RiskEngine(10_000.0).evaluate(
            notional=5_000, signal_score=0.5, realized_vol=0.6, vix=20.0
        )
        assert dec.verdict == Verdict.SCALE
        assert dec.scaled_notional < 5_000

    def test_kelly_zero_when_no_edge(self):
        from risk.engine import RiskEngine
        dec = RiskEngine(10_000.0).evaluate(
            notional=100, signal_score=0.0, realized_vol=0.6, vix=20.0
        )
        assert dec.kelly_optimal == 0.0

    def test_copy_tiers_proportional(self):
        from risk.engine import RiskEngine
        dec = RiskEngine(10_000.0).evaluate(
            notional=100, signal_score=0.3, realized_vol=0.6, vix=20.0
        )
        t = dec.copy_tiers
        assert t is not None
        assert t.auto < t.show_low < t.show_high


# ── QuantMemoryV2 ─────────────────────────────────────────────────────────────

class TestQuantMemoryV2:
    def test_push_and_retrieve(self):
        from scheduler.quant_memory_v2 import QuantMemoryV2, MarketSignalV2
        qm = QuantMemoryV2()
        qm._ring.clear()
        sig = MarketSignalV2(signal_score=0.42, fear_greed=70, regime_idx=1)
        qm.push(sig)
        retrieved = qm.last(1)[0]
        assert abs(retrieved.signal_score - 0.42) < 0.1
        assert retrieved.regime_idx == 1

    def test_ring_never_exceeds_max(self):
        from scheduler.quant_memory_v2 import QuantMemoryV2, MarketSignalV2, RING_SIZE
        qm = QuantMemoryV2()
        qm._ring.clear()
        for i in range(RING_SIZE + 5):
            qm.push(MarketSignalV2(signal_score=float(i) / 20.0))
        assert len(qm._ring) <= RING_SIZE

    def test_bits_used_increases_with_pushes(self):
        from scheduler.quant_memory_v2 import QuantMemoryV2, MarketSignalV2, FIELD_COUNT, BITS
        qm = QuantMemoryV2()
        qm._ring.clear()
        qm.push(MarketSignalV2())
        assert qm.bits_used == FIELD_COUNT * BITS

    def test_summary_returns_string(self):
        from scheduler.quant_memory_v2 import QuantMemoryV2, MarketSignalV2
        qm = QuantMemoryV2()
        qm._ring.clear()
        qm.push(MarketSignalV2(vix_fear=22.0))
        s = qm.summary(1)
        assert isinstance(s, str)
        assert "VIX" in s


# ── TTLCache ──────────────────────────────────────────────────────────────────

class TestTTLCache:
    def test_set_and_get(self):
        from data.cache import TTLCache
        c = TTLCache(maxsize=8, ttl=60.0)
        c.set("k", 99)
        assert c.get("k") == 99

    def test_contains(self):
        from data.cache import TTLCache
        c = TTLCache(maxsize=8, ttl=60.0)
        c.set("x", "hello")
        assert "x" in c
        assert "y" not in c

    def test_ttl_expiry(self):
        import time
        from data.cache import TTLCache
        c = TTLCache(maxsize=8, ttl=0.05)
        c.set("exp", 1)
        time.sleep(0.1)
        assert c.get("exp") is None

    def test_lru_eviction_at_maxsize(self):
        from data.cache import TTLCache
        c = TTLCache(maxsize=3, ttl=60.0)
        c.set("a", 1)
        c.set("b", 2)
        c.set("c", 3)
        c.set("d", 4)
        assert len(c) == 3
        assert c.get("a") is None

    def test_domain_cache_quote(self):
        from data.cache import cache_quote, get_cached_quote
        cache_quote("ETH-USD", {"mid": 3200.0, "spread_bps": 2.0})
        result = get_cached_quote("ETH-USD")
        assert result is not None
        assert result["mid"] == 3200.0


# ── AlphaFactors ──────────────────────────────────────────────────────────────

class TestAlphaFactors:
    def test_compute_returns_dataframe(self):
        from features.alpha_factors import AlphaFactors
        import pandas as pd
        close = np.random.randn(60).cumsum() + 60000
        df = AlphaFactors.compute(close)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 60

    def test_no_inf_values(self):
        from features.alpha_factors import AlphaFactors
        close = np.random.randn(60).cumsum() + 60000
        df = AlphaFactors.compute(close)
        assert not np.isinf(df.values).any()

    def test_rsi_within_0_100(self):
        from features.alpha_factors import AlphaFactors
        close = np.random.randn(60).cumsum() + 60000
        df = AlphaFactors.compute(close)
        rsi = df["alpha_rsi"].values[15:]
        assert (rsi >= 0.0).all() and (rsi <= 100.0).all()

    def test_latest_normalized_tanh_bounded(self):
        from features.alpha_factors import AlphaFactors
        close = np.random.randn(60).cumsum() + 60000
        df    = AlphaFactors.compute(close)
        norms = AlphaFactors.latest_normalized(df)
        for v in norms.values():
            assert -1.0 <= v <= 1.0
