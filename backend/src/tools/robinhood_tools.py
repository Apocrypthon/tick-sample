"""Robinhood tools — read-only portfolio access + gated order execution."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)


# ── auth singleton ────────────────────────────────────────────────────────────

_rh_session: Any = None


def _get_session():
    """Return an authenticated robin_stocks session, logging in on first call."""
    global _rh_session
    if _rh_session is not None:
        return _rh_session
    try:
        import robin_stocks.robinhood as r
    except ImportError:
        raise RuntimeError("robin_stocks not installed. Run: uv add robin_stocks")

    username = os.environ.get("ROBINHOOD_USERNAME", "")
    password = os.environ.get("ROBINHOOD_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("ROBINHOOD_USERNAME and ROBINHOOD_PASSWORD must be set in .env")

    mfa_code = os.environ.get("ROBINHOOD_MFA_CODE", "")
    if mfa_code:
        r.login(username, password, mfa_code=mfa_code)
    else:
        r.login(username, password)

    _rh_session = r
    logger.info("Robinhood session established for %s", username)
    return r


# ── tools ─────────────────────────────────────────────────────────────────────

class GetPortfolioTool(BaseTool):
    name: str = "get_portfolio"
    description: str = "Returns current Robinhood portfolio: holdings, quantities, current price, equity, and P&L."

    def _run(self, *args, **kwargs) -> str:
        r = _get_session()
        # Account-level cash and buying power
        try:
            profile = r.profiles.load_portfolio_profile()
            equity   = profile.get("equity") or profile.get("extended_hours_equity", "?")
            bp       = profile.get("withdrawable_amount", "?")
            lines = [
                "**Robinhood Account**",
                f"• Total equity:    ${equity}",
                f"• Buying power:    ${bp}",
                "",
                "**Holdings**",
            ]
        except Exception:
            lines = ["**Robinhood Holdings**"]
        # Equity holdings
        holdings = r.account.build_holdings()
        if not holdings:
            lines.append("No equity positions.")
        else:
            for symbol, data in holdings.items():
                qty    = data.get("quantity", "?")
                price  = data.get("price", "?")
                equity = data.get("equity", "?")
                pct    = data.get("percent_change", "?")
                pl     = data.get("equity_change", "?")
                lines.append(
                    f"• {symbol}: {qty} shares @ ${price} | "
                    f"equity ${equity} | P&L ${pl} ({pct}%)"
                )
        # Crypto holdings (separate Robinhood endpoint)
        try:
            crypto = r.crypto.get_crypto_positions()
            if crypto:
                lines.append("\n**Robinhood Crypto:**")
                for pos in crypto:
                    currency = pos.get("currency", {}).get("code") or pos.get("currency_code")
                    if not currency or currency == "None":
                        continue
                    qty  = float(pos.get("quantity", 0) or 0)
                    cost = float((pos.get("cost_bases") or [{}])[0].get("direct_cost_basis") or 0)
                    if qty > 0.000001:
                        try:
                            raw_price = r.crypto.get_crypto_quote(currency + "-USD")
                            current_price = float(raw_price.get("mark_price", 0))
                            current_value = qty * current_price
                            pl = current_value - cost
                            lines.append(
                                f"• {currency}: {qty:.6f} @ ${current_price:,.2f} | "
                                f"value ${current_value:.2f} | P&L ${pl:+.2f}"
                            )
                        except Exception:
                            lines.append(f"• {currency}: {qty:.6f}")
        except Exception as e:
            lines.append(f"(Crypto holdings unavailable: {e})")
        return "\n".join(lines)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class GetQuoteTool(BaseTool):
    name: str = "get_quote"
    description: str = "Returns the current market price for a stock symbol. Input: ticker symbol (e.g. AAPL)."

    def _run(self, symbol: str) -> str:
        r = _get_session()
        symbol = symbol.strip().upper()
        price = r.stocks.get_latest_price(symbol)
        if not price or price[0] is None:
            return f"Could not retrieve price for {symbol}."
        return f"{symbol}: ${float(price[0]):.2f}"

    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)


class GetWatchlistTool(BaseTool):
    name: str = "get_watchlist"
    description: str = "Returns symbols currently on the Robinhood watchlist."

    def _run(self, *args, **kwargs) -> str:
        r = _get_session()
        raw = r.account.get_watchlist_by_name("Cryptos to Watch")
        # Handle both flat list and {"results": [...]} dict shapes
        if isinstance(raw, dict):
            items = raw.get("results", [])
        else:
            items = raw or []
        if not items:
            return "Watchlist is empty."
        symbols = []
        for item in items:
            if isinstance(item, str):
                symbols.append(item)
            elif isinstance(item, dict):
                sym = item.get("symbol") or item.get("instrument_url", "").rstrip("/").split("/")[-1]
                if sym:
                    symbols.append(sym)
        if not symbols:
            return f"Watchlist returned {len(items)} items but no symbols could be parsed."
        return "Watchlist: " + ", ".join(symbols)

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)


class PlaceOrderTool(BaseTool):
    name: str = "place_order"
    description: str = (
        "Places a market order on Robinhood. "
        "Inputs: symbol (str), quantity (float), side ('buy' or 'sell'). "
        "TRADE_ENABLED env var must be 'true' or this raises."
    )

    def _run(self, symbol: str, quantity: float, side: str) -> str:
        # ── hard gate — raises, does not return a soft error string ──────────
        if os.environ.get("TRADE_ENABLED", "false").lower() != "true":
            raise RuntimeError(
                "Trading is disabled. Set TRADE_ENABLED=true in .env and re-source to enable."
            )

        r = _get_session()
        symbol = symbol.strip().upper()
        side = side.strip().lower()

        if side == "buy":
            result = r.orders.order_buy_market(symbol, quantity)
        elif side == "sell":
            result = r.orders.order_sell_market(symbol, quantity)
        else:
            raise ValueError(f"side must be 'buy' or 'sell', got: {side}")

        order_id = result.get("id", "unknown")
        state = result.get("state", "unknown")
        return f"Order placed: {side.upper()} {quantity} {symbol} | id={order_id} | state={state}"

    async def _arun(self, symbol: str, quantity: float, side: str) -> str:
        return self._run(symbol, quantity, side)


# ── registration helper ───────────────────────────────────────────────────────

ROBINHOOD_TOOLS: list[BaseTool] = [
    GetPortfolioTool(),
    GetQuoteTool(),
    GetWatchlistTool(),
    PlaceOrderTool(),
]
