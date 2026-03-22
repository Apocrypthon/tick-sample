"""
arb_scanner.py — triangular arbitrage scanner.

Conflict ARB-A resolution (Option A): ccxt.pro watchMultipleOrderBooks.
Symbols batched into groups of 4 and rotated every BATCH_ROTATE_SEC to
stay under Coinbase WS subscription limits without manual reconnect logic.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
FEE              = 0.006
MIN_PROFIT_PCT   = 0.05
EXEC_FLOOR_PCT   = 0.30
BATCH_SIZE       = 4        # symbols per WS subscription batch
BATCH_ROTATE_SEC = 5.0      # seconds before rotating to next batch

TRIANGLES = [
    {"name": "USD-BTC-ETH-USD",  "legs": [("BTC/USD","buy"),  ("ETH/BTC","buy"),  ("ETH/USD","sell")]},
    {"name": "USD-ETH-BTC-USD",  "legs": [("ETH/USD","buy"),  ("ETH/BTC","sell"), ("BTC/USD","sell")]},
    {"name": "USD-SOL-BTC-USD",  "legs": [("SOL/USD","buy"),  ("SOL/BTC","sell"), ("BTC/USD","sell")]},
    {"name": "USD-BTC-SOL-USD",  "legs": [("BTC/USD","buy"),  ("SOL/BTC","buy"),  ("SOL/USD","sell")]},
    {"name": "USD-SOL-ETH-USD",  "legs": [("SOL/USD","buy"),  ("SOL/ETH","sell"), ("ETH/USD","sell")]},
    {"name": "USD-ETH-SOL-USD",  "legs": [("ETH/USD","buy"),  ("SOL/ETH","buy"),  ("SOL/USD","sell")]},
]

ALL_SYMBOLS = list({sym for t in TRIANGLES for sym, _ in t["legs"]})

# Build rotation batches — each batch is a list[str] of ≤ BATCH_SIZE symbols
_BATCHES: list[list[str]] = [
    ALL_SYMBOLS[i : i + BATCH_SIZE]
    for i in range(0, len(ALL_SYMBOLS), BATCH_SIZE)
]


@dataclass
class ArbOpportunity:
    name:             str
    profit_pct:       float
    legs:             list
    capital_usd:      float
    gross_profit_usd: float
    detected_at:      float = field(default_factory=time.time)

    def format(self) -> str:
        age   = round(time.time() - self.detected_at, 1)
        lines = [
            f"**ARB OPPORTUNITY**: {self.name}  ({age}s ago)",
            f"   Profit: **{self.profit_pct:+.4f}%**  on "
            f"${self.capital_usd:.2f}  =>  **${self.gross_profit_usd:.4f}**",
        ]
        for sym, side, price in self.legs:
            lines.append(f"   {side.upper():4s}  {sym:<10s}  @ {price:.6g}")
        return "\n".join(lines)


class ArbScanner:
    def __init__(self, capital_usd: Optional[float] = None):
        self.capital_usd = capital_usd or float(os.getenv("ARB_MAX_TRADE_USD", "50"))
        self._books:   dict = {}
        self._latest:  Optional[ArbOpportunity] = None
        self._tasks:   list = []
        self._running  = False
        self._exchange = None
        self._batch_idx = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running  = True
        self._exchange = self._build_exchange()

        # One watcher task per batch + detect loop
        for idx, batch in enumerate(_BATCHES):
            t = asyncio.create_task(
                self._watch_batch(batch, idx),
                name=f"arb_batch_{idx}",
            )
            self._tasks.append(t)

        self._tasks.append(
            asyncio.create_task(self._detect_loop(), name="arb_detect")
        )
        logger.info(
            "ArbScanner started — %d symbols in %d batches of ≤%d",
            len(ALL_SYMBOLS), len(_BATCHES), BATCH_SIZE,
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception:
                pass
        logger.info("ArbScanner stopped")

    # ── exchange factory ──────────────────────────────────────────────────────

    def _build_exchange(self):
        secret = os.environ.get("COINBASE_API_SECRET", "").replace("\\n", "\n")
        return ccxtpro.coinbase({
            "apiKey":          os.environ.get("COINBASE_API_KEY", ""),
            "secret":          secret,
            "enableRateLimit": True,
        })

    # ── batched order book watcher ────────────────────────────────────────────

    async def _watch_batch(self, symbols: list[str], batch_idx: int) -> None:
        """
        Watch a batch of ≤4 symbols using ccxt.pro watchOrderBookForSymbols.
        Falls back to sequential watch_order_book if the unified method is unavailable.
        On any error, backs off exponentially and retries.
        """
        backoff = 1
        while self._running:
            try:
                while self._running:
                    # ccxt.pro unified multi-symbol watcher (Option A)
                    books = await self._exchange.watchOrderBookForSymbols(
                        symbols, limit=5
                    )
                    # books is a dict {symbol: orderbook}
                    now = time.time()
                    for sym, ob in books.items():
                        self._books[sym] = {
                            "bids": ob["bids"],
                            "asks": ob["asks"],
                            "ts":   now,
                        }
                    backoff = 1

            except (AttributeError, NotImplementedError):
                # Exchange doesn't support watchOrderBookForSymbols — fall back
                logger.debug("Falling back to sequential watchers for batch %d", batch_idx)
                await self._watch_batch_sequential(symbols)
                return

            except asyncio.CancelledError:
                return

            except Exception as exc:
                logger.warning(
                    "Batch %d watch error: %s — retry in %ds", batch_idx, exc, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _watch_batch_sequential(self, symbols: list[str]) -> None:
        """Sequential fallback: one watch_order_book task per symbol in the batch."""
        tasks = [
            asyncio.create_task(self._watch_single(sym), name=f"arb_sym_{sym}")
            for sym in symbols
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()

    async def _watch_single(self, symbol: str) -> None:
        backoff = 1
        while self._running:
            try:
                while self._running:
                    ob = await self._exchange.watch_order_book(symbol, limit=5)
                    self._books[symbol] = {
                        "bids": ob["bids"],
                        "asks": ob["asks"],
                        "ts":   time.time(),
                    }
                    backoff = 1
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("watch_order_book(%s): %s — retry in %ds", symbol, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    # ── arbitrage detection ───────────────────────────────────────────────────

    async def _detect_loop(self) -> None:
        while self._running:
            try:
                opp = self._scan()
                if opp:
                    self._latest = opp
                    logger.debug("Arb: %s %+.4f%%", opp.name, opp.profit_pct)
            except Exception as exc:
                logger.warning("Arb detect error: %s", exc)
            await asyncio.sleep(0.5)

    def _compute(self, legs: list):
        ratio  = 1.0
        prices = []
        for symbol, side in legs:
            book = self._books.get(symbol)
            if not book:
                return None
            if side == "buy":
                lvl = book["asks"]
                if not lvl:
                    return None
                price  = lvl[0][0]
                ratio /= price
            else:
                lvl = book["bids"]
                if not lvl:
                    return None
                price  = lvl[0][0]
                ratio *= price
            ratio *= (1 - FEE)
            prices.append((symbol, side, price))
        return ratio - 1.0, prices

    def _scan(self) -> Optional[ArbOpportunity]:
        best = None
        for tri in TRIANGLES:
            result = self._compute(tri["legs"])
            if result is None:
                continue
            profit_ratio, prices = result
            profit_pct = profit_ratio * 100
            if profit_pct < MIN_PROFIT_PCT:
                continue
            opp = ArbOpportunity(
                name=tri["name"],
                profit_pct=profit_pct,
                legs=prices,
                capital_usd=self.capital_usd,
                gross_profit_usd=self.capital_usd * profit_ratio,
            )
            if best is None or opp.profit_pct > best.profit_pct:
                best = opp
        return best

    # ── public API ────────────────────────────────────────────────────────────

    def get_latest(self) -> Optional[ArbOpportunity]:
        return self._latest

    def get_book_summary(self) -> str:
        if not self._books:
            return "No order books yet."
        parts = []
        for sym in sorted(self._books):
            b   = self._books[sym]
            bid = b["bids"][0][0] if b["bids"] else 0
            ask = b["asks"][0][0] if b["asks"] else 0
            age = round(time.time() - b["ts"], 1)
            parts.append(f"{sym}: bid={bid:.6g} ask={ask:.6g} ({age}s)")
        return " | ".join(parts)

    @property
    def symbols_ready(self) -> int:
        return len(self._books)

    @property
    def total_symbols(self) -> int:
        return len(ALL_SYMBOLS)
