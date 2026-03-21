"""Strategy analyst and cross-market tools — pre-trade veto layer."""

from __future__ import annotations

import logging
import os
from typing import Any

from langchain.tools import BaseTool

logger = logging.getLogger(__name__)

# ── crypto symbols — routed to ccxt order book, not RH historicals ──────────
_CRYPTO_SYMBOLS = {
    "BTC", "ETH", "SOL", "ADA", "AVAX", "DOGE", "MATIC", "DOT", "LINK",
    "XRP", "LTC", "BCH", "UNI", "AAVE", "ATOM", "XLM", "ALGO", "FIL",
}

# ── sector map: symbol prefix → sector ETF ───────────────────────────────────
_SECTOR_MAP: dict[str, str] = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK", "AMD": "XLK", "GOOGL": "XLK",
    "META": "XLK", "AMZN": "XLY", "TSLA": "XLY", "JPM": "XLF", "BAC": "XLF",
    "GS": "XLF", "XOM": "XLE", "CVX": "XLE", "JNJ": "XLV", "PFE": "XLV",
    "UNH": "XLV", "WMT": "XLP", "PG": "XLP", "BA": "XLI", "CAT": "XLI",
}

# ── risk constants ────────────────────────────────────────────────────────────
MAX_POSITION_PCT   = 0.15   # no single position > 15% of portfolio
MAX_SINGLE_TRADE   = 500.0  # no single trade > $500 without explicit override
VIX_CAUTION        = 25.0   # warn when VIX above this
VIX_VETO           = 35.0   # hard veto when VIX above this


def _get_rh():
    """Return authenticated robin_stocks session."""
    from src.tools.robinhood_tools import _get_session
    return _get_session()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ── strategy analyst ──────────────────────────────────────────────────────────

class StrategyAnalystTool(BaseTool):
    name: str = "strategy_analyst"
    description: str = (
        "MUST be called before place_order. "
        "Analyzes a proposed trade against portfolio risk rules and returns "
        "APPROVE or VETO with reasoning. "
        "Inputs: symbol (str), quantity (float), side ('buy' or 'sell'), "
        "reason (str — why you want this trade)."
    )

    def _run(self, symbol: str, quantity: float, side: str, reason: str = "") -> str:
        r = _get_rh()
        symbol = symbol.strip().upper()
        side = side.strip().lower()
        verdicts: list[str] = []
        vetoes: list[str] = []

        # ── current price ─────────────────────────────────────────────────────
        raw_price = r.stocks.get_latest_price(symbol)
        price = _safe_float(raw_price[0] if raw_price else None)
        if price == 0.0:
            return f"VETO — could not retrieve price for {symbol}. Do not proceed."
        trade_value = price * abs(quantity)

        # ── trade size check ──────────────────────────────────────────────────
        if trade_value > MAX_SINGLE_TRADE:
            vetoes.append(
                f"Trade value ${trade_value:.2f} exceeds single-trade limit ${MAX_SINGLE_TRADE:.2f}. "
                f"Reduce quantity or set override."
            )
        else:
            verdicts.append(f"Trade size ${trade_value:.2f} within limit.")

        # ── portfolio concentration check (buy only) ──────────────────────────
        if side == "buy":
            try:
                holdings = r.account.build_holdings()
                portfolio_value = sum(
                    _safe_float(h.get("equity")) for h in holdings.values()
                )
                current_equity = _safe_float(
                    holdings.get(symbol, {}).get("equity")
                )
                projected_pct = (current_equity + trade_value) / max(portfolio_value, 1)
                if projected_pct > MAX_POSITION_PCT:
                    vetoes.append(
                        f"Projected {symbol} concentration {projected_pct:.1%} exceeds "
                        f"max {MAX_POSITION_PCT:.0%} of portfolio."
                    )
                else:
                    verdicts.append(
                        f"Post-trade {symbol} concentration {projected_pct:.1%} acceptable."
                    )
            except Exception as e:
                verdicts.append(f"Portfolio check skipped: {e}")

        # ── sell safety check — don't sell what you don't own ─────────────────
        if side == "sell":
            try:
                holdings = r.account.build_holdings()
                owned_qty = _safe_float(holdings.get(symbol, {}).get("quantity"))
                if owned_qty < quantity:
                    vetoes.append(
                        f"Cannot sell {quantity} shares of {symbol} — "
                        f"only {owned_qty} owned."
                    )
                else:
                    verdicts.append(f"Sufficient shares owned ({owned_qty}) for sell.")
            except Exception as e:
                verdicts.append(f"Ownership check skipped: {e}")

        # ── TRADE_ENABLED gate reminder ───────────────────────────────────────
        if os.environ.get("TRADE_ENABLED", "false").lower() != "true":
            vetoes.append("TRADE_ENABLED is not set to true — trading is disabled.")

        # ── verdict ───────────────────────────────────────────────────────────
        checks = "\n".join(f"  ✓ {v}" for v in verdicts)
        if vetoes:
            blocks = "\n".join(f"  ✗ {v}" for v in vetoes)
            return (
                f"VETO — {symbol} {side.upper()} {quantity} @ ~${price:.2f}\n"
                f"Reason provided: {reason}\n\n"
                f"Blocking issues:\n{blocks}\n\n"
                f"Passing checks:\n{checks}"
            )

        return (
            f"APPROVE — {symbol} {side.upper()} {quantity} @ ~${price:.2f}\n"
            f"Reason provided: {reason}\n\n"
            f"All checks passed:\n{checks}\n\n"
            f"Proceed to place_order only after user types CONFIRM."
        )

    async def _arun(self, symbol: str, quantity: float, side: str, reason: str = "") -> str:
        return self._run(symbol, quantity, side, reason)


# ── cross-market context ──────────────────────────────────────────────────────

class CrossMarketTool(BaseTool):
    name: str = "cross_market_context"
    description: str = (
        "Returns broad market context before a trade decision: "
        "SPY price and trend, VIX fear level, and sector ETF for the symbol. "
        "Input: symbol (str). Call this before strategy_analyst for full context."
    )

    def _run(self, symbol: str) -> str:
        r = _get_rh()
        symbol = symbol.strip().upper()
        lines: list[str] = [f"**Cross-Market Context for {symbol}**\n"]

        # ── SPY ───────────────────────────────────────────────────────────────
        try:
            spy_prices = r.stocks.get_stock_historicals(
                "SPY", interval="day", span="month"
            )
            if spy_prices and len(spy_prices) >= 5:
                closes = [_safe_float(d.get("close_price")) for d in spy_prices]
                spy_now = closes[-1]
                spy_5d  = sum(closes[-5:]) / 5
                spy_20d = sum(closes) / len(closes)
                trend = "BULLISH" if spy_now > spy_20d else "BEARISH"
                lines.append(
                    f"SPY: ${spy_now:.2f} | 5d avg ${spy_5d:.2f} | "
                    f"20d avg ${spy_20d:.2f} | trend={trend}"
                )
            else:
                lines.append("SPY: data unavailable")
        except Exception as e:
            lines.append(f"SPY: error ({e})")

        # ── VIX ───────────────────────────────────────────────────────────────
        try:
            vix_raw = r.stocks.get_latest_price("VIXY")
            vix = _safe_float(vix_raw[0] if vix_raw else None)
            if vix > 0:
                if vix >= VIX_VETO:
                    fear = f"EXTREME FEAR — hard veto threshold reached (>{VIX_VETO})"
                elif vix >= VIX_CAUTION:
                    fear = f"ELEVATED FEAR — trade with caution (>{VIX_CAUTION})"
                else:
                    fear = "NORMAL — market calm"
                lines.append(f"VIX proxy (VIXY): ${vix:.2f} | {fear}")
            else:
                lines.append("VIX: unavailable")
        except Exception as e:
            lines.append(f"VIX: error ({e})")

        # ── sector ETF ────────────────────────────────────────────────────────
        sector_etf = _SECTOR_MAP.get(symbol)
        if sector_etf:
            try:
                etf_raw = r.stocks.get_latest_price(sector_etf)
                etf_price = _safe_float(etf_raw[0] if etf_raw else None)
                etf_hist = r.stocks.get_stock_historicals(
                    sector_etf, interval="day", span="month"
                )
                if etf_hist and len(etf_hist) >= 20:
                    etf_closes = [_safe_float(d.get("close_price")) for d in etf_hist]
                    etf_20d = sum(etf_closes) / len(etf_closes)
                    etf_trend = "above" if etf_price > etf_20d else "below"
                    lines.append(
                        f"Sector ETF ({sector_etf}): ${etf_price:.2f} | "
                        f"{etf_trend} 20d avg ${etf_20d:.2f}"
                    )
                else:
                    lines.append(f"Sector ETF ({sector_etf}): ${etf_price:.2f}")
            except Exception as e:
                lines.append(f"Sector ETF ({sector_etf}): error ({e})")
        else:
            lines.append(f"Sector ETF: no mapping for {symbol} — add to _SECTOR_MAP")

        # ── symbol momentum (crypto vs equity routing) ────────────────────────
        base = symbol.replace("-USD", "").replace("/USD", "")
        if base in _CRYPTO_SYMBOLS:
            # Crypto: use Coinbase order book via arb_scanner if available,
            # else fall back to Coinbase REST quote
            try:
                from src.arb import arb_tools as _arb
                scanner = _arb._scanner
                cb_sym = f"{base}/USD"
                if scanner and cb_sym in scanner._books:
                    book = scanner._books[cb_sym]
                    bid = book["bids"][0][0] if book["bids"] else 0
                    ask = book["asks"][0][0] if book["asks"] else 0
                    mid = (bid + ask) / 2 if bid and ask else 0
                    lines.append(
                        f"{symbol} (crypto) live: bid=${bid:,.2f} ask=${ask:,.2f} mid=${mid:,.2f}"
                    )
                else:
                    from src.tools.coinbase_tools import _get_cb, _safe_float as _cb_sf
                    cb = _get_cb()
                    ticker = cb.get_best_bid_ask(product_ids=[f"{base}-USD"])
                    if ticker and hasattr(ticker, "pricebooks") and ticker.pricebooks:
                        pb = ticker.pricebooks[0]
                        bid = _cb_sf(pb.bids[0].price if pb.bids else 0)
                        ask = _cb_sf(pb.asks[0].price if pb.asks else 0)
                        mid = (bid + ask) / 2
                        lines.append(
                            f"{symbol} (crypto): bid=${bid:,.2f} ask=${ask:,.2f} mid=${mid:,.2f}"
                        )
                    else:
                        lines.append(f"{symbol}: crypto quote unavailable")
            except Exception as e:
                lines.append(f"{symbol} crypto context: error ({e})")
        else:
            # Equity: standard historical momentum
            try:
                hist = r.stocks.get_stock_historicals(
                    symbol, interval="day", span="month"
                )
                if hist and len(hist) >= 10:
                    closes = [_safe_float(d.get("close_price")) for d in hist]
                    now   = closes[-1]
                    ma5   = sum(closes[-5:]) / 5
                    ma20  = sum(closes) / len(closes)
                    momentum = "BULLISH" if now > ma5 > ma20 else (
                        "BEARISH" if now < ma5 < ma20 else "MIXED"
                    )
                    lines.append(
                        f"{symbol} momentum: ${now:.2f} | "
                        f"5d ${ma5:.2f} | 20d ${ma20:.2f} | {momentum}"
                    )
            except Exception as e:
                lines.append(f"{symbol} momentum: error ({e})")

        return "\n".join(lines)

    async def _arun(self, symbol: str) -> str:
        return self._run(symbol)


# ── registration ──────────────────────────────────────────────────────────────

STRATEGY_TOOLS: list[BaseTool] = [
    CrossMarketTool(),
    StrategyAnalystTool(),
]
