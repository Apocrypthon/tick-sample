"""execution.py — routes approved trade to the correct venue."""
from __future__ import annotations

import asyncio
import logging
import os

from langchain_core.runnables import RunnableConfig

from src.agents.trading.state import TradingState

logger = logging.getLogger(__name__)


async def run_execution(state: TradingState, config: RunnableConfig) -> dict:
    """Execute an AUTO-tier trade. Only called after Discord ✅ reaction."""
    plan = state.get("execution_plan")
    if not plan:
        return {"execution_result": "No execution plan.", "phase": "done"}
    if not state.get("approval_granted"):
        return {"execution_result": "Not approved — skipping.", "phase": "done"}
    if os.environ.get("TRADE_ENABLED", "false").lower() != "true":
        msg = (
            f"[DRY RUN] Would execute: {plan['side']} "
            f"${plan['copy_trade_auto']:.2f} of {plan['symbol']} on {plan['venue']}. "
            "Set TRADE_ENABLED=true to go live."
        )
        logger.info(msg)
        return {"execution_result": msg, "phase": "done"}

    notional = plan.get("copy_trade_auto", 0.0)
    if notional <= 0:
        return {"execution_result": "AUTO tier notional is $0 — skipping.", "phase": "done"}

    venue = plan.get("venue", "alpaca")
    symbol = plan.get("symbol", "")
    side = plan.get("side", "BUY")

    try:
        if venue in ("coinbase", "both"):
            result = await _exec_coinbase(symbol, notional, side)
        else:
            result = await _exec_alpaca(symbol, notional, side)
        logger.info("Execution OK: %s", result)
        return {"execution_result": result, "phase": "done"}
    except Exception as exc:
        logger.exception("Execution failed")
        return {"execution_result": f"Execution error: {exc}", "execution_error": str(exc), "phase": "done"}


async def _exec_coinbase(symbol: str, notional: float, side: str) -> str:
    from src.tools.coinbase_tools import PlaceCryptoOrderTool
    return await asyncio.to_thread(PlaceCryptoOrderTool()._run, symbol, notional, side)


async def _exec_alpaca(symbol: str, notional: float, side: str) -> str:
    from src.tools.alpaca_tools import PlaceAlpacaOrderTool
    return await asyncio.to_thread(PlaceAlpacaOrderTool()._run, symbol, notional, side)
