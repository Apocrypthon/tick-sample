"""hydrate.py — pre-analyst data enrichment: live quotes + portfolio snapshots."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from langchain_core.runnables import RunnableConfig

from src.agents.trading.state import TradingState

logger = logging.getLogger(__name__)


def _symbols_from_trigger(trigger: dict) -> list[str]:
    """Extract tradeable symbols from trigger context."""
    t = trigger.get("type", "")
    symbols: list[str] = []
    if t == "threshold":
        sym = trigger.get("symbol", "")
        if sym:
            symbols.append(sym)
    elif t == "arb":
        opp = trigger.get("opportunity", {})
        for leg in opp.get("legs", []):
            s = leg.get("symbol", "") if isinstance(leg, dict) else str(leg)
            if s:
                symbols.append(s)
    # Macro anchors — always include for cross-asset context
    for anchor in ("BTC", "ETH"):
        if anchor not in symbols:
            symbols.append(anchor)
    return [s for s in symbols if s]


async def _fetch_quotes(symbols: list[str]) -> dict[str, dict]:
    """Fetch bid/ask/mid from Coinbase for each symbol."""
    quotes: dict[str, dict] = {}
    if not symbols:
        return quotes
    try:
        from src.tools.coinbase_tools import _get_cb, _safe_float

        cb = await asyncio.to_thread(_get_cb)
        product_ids = [f"{s}-USD" if "-" not in s else s for s in symbols]
        ticker = await asyncio.to_thread(cb.get_best_bid_ask, product_ids=product_ids)
        if ticker and hasattr(ticker, "pricebooks"):
            for pb in ticker.pricebooks:
                pid = getattr(pb, "product_id", "")
                bid = _safe_float(pb.bids[0].price) if pb.bids else 0.0
                ask = _safe_float(pb.asks[0].price) if pb.asks else 0.0
                mid = (bid + ask) / 2 if bid and ask else 0.0
                spread_bps = ((ask - bid) / mid * 10_000) if mid else 0.0
                quotes[pid] = {
                    "bid": bid, "ask": ask, "mid": mid,
                    "spread_bps": round(spread_bps, 1),
                }
    except Exception as exc:
        logger.warning("Quote fetch failed (non-fatal): %s", exc)
    return quotes


async def _fetch_portfolios() -> dict[str, Any]:
    """Snapshot portfolios from both venues in parallel."""
    result: dict[str, Any] = {}

    async def _coinbase():
        try:
            from src.tools.coinbase_tools import GetCryptoPortfolioTool
            result["coinbase"] = await asyncio.to_thread(GetCryptoPortfolioTool()._run)
        except Exception as exc:
            result["coinbase"] = f"[unavailable: {exc}]"

    async def _alpaca():
        try:
            from src.tools.alpaca_tools import GetAlpacaPortfolioTool
            result["alpaca"] = await asyncio.to_thread(GetAlpacaPortfolioTool()._run)
        except Exception as exc:
            result["alpaca"] = f"[unavailable: {exc}]"

    await asyncio.gather(_coinbase(), _alpaca())
    return result


async def run_hydrate(state: TradingState, config: RunnableConfig) -> dict:
    """Fetch live market data + portfolio state before analysts run."""
    trigger = state.get("trigger") or {}
    symbols = _symbols_from_trigger(trigger)

    t0 = time.time()
    quotes, portfolios = await asyncio.gather(
        _fetch_quotes(symbols),
        _fetch_portfolios(),
    )
    elapsed = time.time() - t0

    market_data = {
        "quotes": quotes,
        "portfolios": portfolios,
        "symbols": symbols,
        "fetched_at": time.time(),
    }

    logger.info(
        "Hydrate: %d quotes, %d venues in %.1fs",
        len(quotes), len(portfolios), elapsed,
    )
    return {"market_data": market_data, "phase": "analysis"}
