"""Coinbase Advanced Trade tools — crypto quotes, portfolio, and gated execution."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

_cb_client = None


def _get_cb():
    global _cb_client
    if _cb_client is not None:
        return _cb_client
    try:
        from coinbase.rest import RESTClient
    except ImportError:
        raise RuntimeError(
            "coinbase-advanced-py not installed. Run: uv add coinbase-advanced-py"
        )

    api_key = os.environ.get("COINBASE_API_KEY", "")
    api_secret = os.environ.get("COINBASE_API_SECRET", "").replace("\\n", "\n")

    if not api_key or not api_secret:
        raise RuntimeError(
            "COINBASE_API_KEY and COINBASE_API_SECRET must be set in .env"
        )

    _cb_client = RESTClient(api_key=api_key, api_secret=api_secret)
    logger.info("Coinbase Advanced Trade client initialized")
    return _cb_client


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_crypto_usd_value(cb: Any, currency: str, balance: float) -> float:
    try:
        ticker = cb.get_best_bid_ask(product_ids=[f"{currency}-USD"])
        if ticker and hasattr(ticker, "pricebooks") and ticker.pricebooks:
            pb = ticker.pricebooks[0]
            if pb.bids and pb.asks:
                mid = (
                    _safe_float(pb.bids[0].price) + _safe_float(pb.asks[0].price)
                ) / 2
                return balance * mid
    except Exception:
        pass
    return 0.0


# ── tools ─────────────────────────────────────────────────────────────────────


class GetCryptoQuoteTool(BaseTool):
    name: str = "get_crypto_quote"
    description: str = (
        "Returns current bid, ask, and spot price for a Coinbase product. "
        "Input: product_id (str) e.g. 'BTC-USD', 'ETH-USD', 'SOL-USD'."
    )

    def _run(self, product_id: str) -> str:
        cb = _get_cb()
        product_id = product_id.strip().upper()
        try:
            ticker = cb.get_best_bid_ask(product_ids=[product_id])
            if ticker and hasattr(ticker, "pricebooks") and ticker.pricebooks:
                pb = ticker.pricebooks[0]
                bid = _safe_float(pb.bids[0].price if pb.bids else None)
                ask = _safe_float(pb.asks[0].price if pb.asks else None)
                mid = (bid + ask) / 2 if bid and ask else 0.0
                return f"{product_id}: bid=${bid:.2f} ask=${ask:.2f} mid=${mid:.2f}"
            product = cb.get_product(product_id)
            price = _safe_float(getattr(product, "price", None))
            return f"{product_id}: ${price:.2f}"
        except Exception as e:
            return f"Error fetching {product_id}: {e}"

    async def _arun(self, product_id: str) -> str:
        return self._run(product_id)


class GetCryptoPortfolioTool(BaseTool):
    name: str = "get_crypto_portfolio"
    description: str = (
        "Returns Coinbase Advanced Trade account balances and portfolio value."
    )

    def _run(self, *args, **kwargs) -> str:
        cb = _get_cb()
        try:
            accounts = cb.get_accounts()
            lines = ["**Coinbase Portfolio**\n"]
            total_usd = 0.0
            crypto_lines = []
            if hasattr(accounts, "accounts"):
                for acct in accounts.accounts:
                    av = getattr(acct, "available_balance", None)
                    if isinstance(av, dict):
                        balance = _safe_float(av.get("value"))
                    else:
                        balance = _safe_float(getattr(av, "value", None))
                    currency = getattr(acct, "currency", "?")
                    if balance <= 0.0001:
                        continue
                    if currency in ("USD", "USDC", "USDT"):
                        total_usd += balance
                        lines.append(f"• {currency}: ${balance:,.2f}")
                    else:
                        # Try to get USD value via quote
                        usd_val = _get_crypto_usd_value(cb, currency, balance)
                        total_usd += usd_val
                        if usd_val > 0:
                            crypto_lines.append(
                                f"• {currency}: {balance:.6f}  (~${usd_val:,.2f} USD)"
                            )
                        else:
                            crypto_lines.append(f"• {currency}: {balance:.6f}")
            lines.extend(crypto_lines)
            lines.append(f"\nEstimated total USD value: ${total_usd:,.2f}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error fetching Coinbase portfolio: {e}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run()


class PlaceCryptoOrderTool(BaseTool):
    name: str = "place_crypto_order"
    description: str = (
        "Places a market order on Coinbase Advanced Trade. "
        "Inputs: product_id (str e.g. 'BTC-USD'), quote_size (float, USD amount), "
        "side ('BUY' or 'SELL'). "
        "TRADE_ENABLED must be 'true' or this raises. "
        "strategy_analyst must APPROVE before calling this."
    )

    def _run(self, product_id: str, quote_size: float, side: str) -> str:
        if os.environ.get("TRADE_ENABLED", "false").lower() != "true":
            raise RuntimeError(
                "Trading disabled. Set TRADE_ENABLED=true in .env to enable."
            )
        cb = _get_cb()
        import uuid

        product_id = product_id.strip().upper()
        side = side.strip().upper()
        client_order_id = str(uuid.uuid4())

        try:
            if side == "BUY":
                result = cb.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=str(quote_size),
                )
            elif side == "SELL":
                result = cb.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size=str(quote_size),
                )
            else:
                raise ValueError(f"side must be BUY or SELL, got: {side}")

            order = getattr(result, "order", None) or result
            order_id = getattr(order, "order_id", "unknown")
            status = getattr(order, "status", "unknown")
            return (
                f"Coinbase order placed: {side} ${quote_size} of {product_id} | "
                f"order_id={order_id} | status={status}"
            )
        except Exception as e:
            return f"Order failed: {e}"

    async def _arun(self, product_id: str, quote_size: float, side: str) -> str:
        return self._run(product_id, quote_size, side)


COINBASE_TOOLS: list[BaseTool] = [
    GetCryptoQuoteTool(),
    GetCryptoPortfolioTool(),
    PlaceCryptoOrderTool(),
]
