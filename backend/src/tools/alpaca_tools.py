"""Alpaca paper trading tools — parallel simulated execution."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

_alpaca_client = None


def _get_alpaca():
    global _alpaca_client
    if _alpaca_client is not None:
        return _alpaca_client
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        raise RuntimeError("alpaca-py not installed. Run: uv add alpaca-py")

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")

    _alpaca_client = TradingClient(api_key, secret_key, paper=paper)
    mode = "PAPER" if paper else "LIVE"
    logger.info("Alpaca %s client initialized", mode)
    return _alpaca_client


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── tools ─────────────────────────────────────────────────────────────────────

class GetAlpacaPortfolioTool(BaseTool):
    name: str = "get_alpaca_portfolio"
    description: str = (
        "Returns Alpaca paper portfolio: account equity, buying power, "
        "positions and unrealized P&L. Always paper — safe to call anytime."
    )

    def _run(self, *args, **kwargs) -> str:
        client = _get_alpaca()
        try:
            account = client.get_account()
            equity = _safe_float(getattr(account, "equity", 0))
            buying_power = _safe_float(getattr(account, "buying_power", 0))
            cash = _safe_float(getattr(account, "cash", 0))
            unrealized_pl = _safe_float(getattr(account, "unrealized_pl", None) or getattr(account, "unrealized_plpc", None) or 0)
            lines = [
                "**Alpaca Paper Portfolio**\n",
                f"• Equity: ${equity:,.2f}",
                f"• Cash: ${cash:,.2f}",
                f"• Buying power: ${buying_power:,.2f}",
                f"• Unrealized P&L: ${unrealized_pl:+,.2f}",
            ]
            from alpaca.trading.requests import GetAssetsRequest
            positions = client.get_all_positions()
            if positions:
                lines.append("\n**Open Positions:**")
                for pos in positions:
                    sym = pos.symbol
                    qty = _safe_float(pos.qty)
                    pl = _safe_float(pos.unrealized_pl)
                    pct = _safe_float(pos.unrealized_plpc) * 100
                    lines.append(f"• {sym}: {qty} @ ${_safe_float(pos.avg_entry_price):.2f} | P&L ${pl:+.2f} ({pct:+.2f}%)")
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching Alpaca portfolio: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run()


class PlaceAlpacaOrderTool(BaseTool):
    name: str = "place_alpaca_order"
    description: str = (
        "Places a PAPER market order on Alpaca. Always simulated — no real money ever. "
        "No TRADE_ENABLED gate — paper execution is always permitted. "
        "Inputs: symbol (str e.g. 'BTC/USD'), notional (float, USD amount), "
        "side ('buy' or 'sell'). "
        "Use this in parallel with place_crypto_order to compare real vs paper P&L."
    )

    def _run(self, symbol: str, notional: float, side: str) -> str:
        # Paper trading — no TRADE_ENABLED gate needed
        client = _get_alpaca()
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        symbol = symbol.strip().upper()
        side = side.strip().lower()

        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            order_data = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=order_side,
                time_in_force=TimeInForce.GTC,
            )
            order = client.submit_order(order_data=order_data)
            return (
                f"Alpaca PAPER order: {side.upper()} ${notional:.2f} of {symbol} | "
                f"id={order.id} | status={order.status}"
            )
        except Exception as e:
            return f"Alpaca order failed: {e}"

    async def _arun(self, symbol: str, notional: float, side: str) -> str:
        return self._run(symbol, notional, side)


ALPACA_TOOLS: list[BaseTool] = [
    GetAlpacaPortfolioTool(),
    PlaceAlpacaOrderTool(),
]
