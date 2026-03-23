"""hydrate_v2.py — enriched hydrate node with signal engine + market intelligence."""
import sys as _sys, pathlib as _pl
_src = str(_pl.Path(__file__).resolve().parents[4] / "src")
if _src not in _sys.path:
    _sys.path.insert(0, _src)
del _sys, _pl, _src

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.agents.trading.state import TradingState
from src.data.cache import (
    cache_quote, get_cached_quote,
    cache_portfolio, get_cached_portfolio,
)

logger = logging.getLogger(__name__)

# Module-level engine caches (persist across DAG invocations)
_signal_engines: dict = {}
_market_intel   = None


def _symbols_from_trigger(trigger: dict) -> list[str]:
    t, symbols = trigger.get("type", ""), []
    if t == "threshold":
        sym = trigger.get("symbol", "")
        if sym:
            symbols.append(sym)
    elif t == "arb":
        for leg in trigger.get("opportunity", {}).get("legs", []):
            s = leg.get("symbol", "") if isinstance(leg, dict) else str(leg)
            if s:
                symbols.append(s)
    for anchor in ("BTC", "ETH"):
        if anchor not in symbols:
            symbols.append(anchor)
    return [s for s in symbols if s]


async def _fetch_quotes(symbols: list[str]) -> dict[str, dict]:
    quotes: dict[str, dict] = {}
    try:
        from src.tools.coinbase_tools import _get_cb, _safe_float
        cb         = await asyncio.to_thread(_get_cb)
        prod_ids   = [f"{s}-USD" if "-" not in s else s for s in symbols]
        ticker     = await asyncio.to_thread(cb.get_best_bid_ask, product_ids=prod_ids)
        if ticker and hasattr(ticker, "pricebooks"):
            for pb in ticker.pricebooks:
                pid  = getattr(pb, "product_id", "")
                bid  = _safe_float(pb.bids[0].price) if pb.bids else 0.0
                ask  = _safe_float(pb.asks[0].price) if pb.asks else 0.0
                mid  = (bid + ask) / 2 if bid and ask else 0.0
                spbps = (ask - bid) / mid * 10_000 if mid else 0.0
                quotes[pid] = {"bid": bid, "ask": ask, "mid": mid,
                               "spread_bps": round(spbps, 1)}
    except Exception as exc:
        logger.warning("Quote fetch failed (non-fatal): %s", exc)
    return quotes


async def _fetch_portfolios() -> dict[str, Any]:
    result: dict[str, Any] = {}

    async def _cb():
        try:
            from src.tools.coinbase_tools import GetCryptoPortfolioTool
            result["coinbase"] = await asyncio.to_thread(GetCryptoPortfolioTool()._run)
        except Exception as exc:
            result["coinbase"] = f"[unavailable: {exc}]"

    async def _alp():
        try:
            from src.tools.alpaca_tools import GetAlpacaPortfolioTool
            result["alpaca"] = await asyncio.to_thread(GetAlpacaPortfolioTool()._run)
        except Exception as exc:
            result["alpaca"] = f"[unavailable: {exc}]"

    await asyncio.gather(_cb(), _alp())
    return result


def _build_signal_context(quotes: dict, vix: float) -> dict[str, Any]:
    global _signal_engines, _market_intel
    ctx: dict[str, Any] = {"vix": vix, "signals": {}, "narrative": "", "regime": "UNKNOWN"}

    try:
        from src.signals.signal_engine import SignalEngine
        from src.intelligence.market_intelligence import MarketIntelligence

        prices = {pid: q["mid"] for pid, q in quotes.items() if q.get("mid")}
        if not prices:
            return ctx

        for pid, q in quotes.items():
            if not q.get("mid"):
                continue
            if pid not in _signal_engines:
                _signal_engines[pid] = SignalEngine(pid)
            sig = _signal_engines[pid].update(
                q["mid"], bid=q.get("bid", 0.0), ask=q.get("ask", 0.0)
            )
            if sig:
                ctx["signals"][pid] = {
                    "score":            round(sig.score, 4),
                    "rsi":              round(sig.rsi, 1),
                    "z":                round(sig.z_score, 3),
                    "vol_regime":       sig.vol_regime,
                    "kalman_forecast":  round(sig.kalman.forecast_5, 4),
                    "kalman_velocity":  round(sig.kalman.velocity, 6),
                }

        if _market_intel is None:
            _market_intel = MarketIntelligence()
        snap = _market_intel.update(prices, vix=vix)
        if snap:
            ctx["regime"]     = snap.regime
            ctx["fear_greed"] = snap.fear_greed
            ctx["narrative"]  = snap.narrative

    except Exception as exc:
        logger.warning("Signal context build failed (non-fatal): %s", exc)

    return ctx


async def run_hydrate(state: TradingState, config: RunnableConfig) -> dict:
    trigger = state.get("trigger") or {}
    symbols = _symbols_from_trigger(trigger)
    vix     = float(trigger.get("vix", 20.0))

    t0 = time.time()
    quotes, portfolios = await asyncio.gather(
        _fetch_quotes(symbols),
        _fetch_portfolios(),
    )
    elapsed = time.time() - t0

    signal_context = _build_signal_context(quotes, vix)

    logger.info(
        "Hydrate v2: %d quotes, %d venues in %.1fs | signals=%d regime=%s",
        len(quotes), len(portfolios), elapsed,
        len(signal_context.get("signals", {})),
        signal_context.get("regime", "?"),
    )

    return {
        "market_data":    {"quotes": quotes, "portfolios": portfolios,
                           "symbols": symbols, "fetched_at": time.time()},
        "signal_context": signal_context,
        "phase":          "analysis",
    }
