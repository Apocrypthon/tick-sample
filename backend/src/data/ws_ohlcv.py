"""ws_ohlcv.py — real-time 1-minute OHLCV candle aggregator via ccxt.pro."""
from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import ccxt.pro as ccxtpro

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Candle:
    symbol:   str
    ts_open:  int
    open:     float
    high:     float
    low:      float
    close:    float
    volume:   float
    complete: bool

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass
class CandleBuffer:
    MAX_BARS = 512
    symbol:  str
    bars:    deque[Candle] = field(default_factory=lambda: deque(maxlen=512))

    def push(self, candle: Candle) -> None:
        self.bars.append(candle)

    def latest(self, n: int = 1) -> list[Candle]:
        return list(self.bars)[-n:]

    def closes(self, n: int = 60) -> list[float]:
        return [c.close for c in list(self.bars)[-n:]]

    def volumes(self, n: int = 60) -> list[float]:
        return [c.volume for c in list(self.bars)[-n:]]

    def __len__(self) -> int:
        return len(self.bars)


class OHLCVAggregator:
    TIMEFRAME = "1m"

    def __init__(
        self,
        symbols:   list[str] | None = None,
        on_candle: Optional[Callable[[Candle], None]] = None,
    ):
        self._symbols   = symbols or ["BTC/USD", "ETH/USD", "SOL/USD"]
        self._on_candle = on_candle
        self._buffers:  dict[str, CandleBuffer] = {
            s: CandleBuffer(s) for s in self._symbols
        }
        self._tasks:    list[asyncio.Task] = []
        self._running   = False
        self._exchange  = None
        self._last_ts:  dict[str, int] = {}

    async def start(self) -> None:
        if self._running:
            return
        self._running  = True
        self._exchange = self._build_exchange()
        for sym in self._symbols:
            t = asyncio.create_task(
                self._watch_symbol(sym),
                name=f"ohlcv_{sym.replace('/', '_')}",
            )
            self._tasks.append(t)
        logger.info("OHLCVAggregator started — %d symbols @ %s",
                    len(self._symbols), self.TIMEFRAME)

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._exchange:
            try:
                await self._exchange.close()
            except Exception:
                pass
        logger.info("OHLCVAggregator stopped")

    def _build_exchange(self):
        secret = os.environ.get("COINBASE_API_SECRET", "").replace("\\n", "\n")
        return ccxtpro.coinbase({
            "apiKey":          os.environ.get("COINBASE_API_KEY", ""),
            "secret":          secret,
            "enableRateLimit": True,
        })

    async def _watch_symbol(self, symbol: str) -> None:
        """Poll 1m OHLCV bars via REST fetch_ohlcv (Coinbase does not support watchOHLCV)."""
        backoff = 5
        last_ts  = 0
        while self._running:
            try:
                # fetch last 3 closed 1m bars (REST — works on all ccxt exchanges)
                candles = await asyncio.to_thread(
                    self._exchange.fetch_ohlcv, symbol, "1m", limit=3
                )
                if candles:
                    new_candles = [c for c in candles if c[0] > last_ts]
                    if new_candles:
                        self._process_bars(symbol, new_candles)
                        last_ts = new_candles[-1][0]
                await asyncio.sleep(60)   # poll once per minute
                backoff = 5
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("OHLCVAggregator[%s] error: %s — retry in %ds", symbol, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)
