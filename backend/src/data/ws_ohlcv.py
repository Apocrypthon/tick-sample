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
        backoff = 1
        while self._running:
            try:
                while self._running:
                    ohlcv = await self._exchange.watchOHLCV(symbol, self.TIMEFRAME)
                    self._process_bars(symbol, ohlcv)
                    backoff = 1
            except asyncio.CancelledError:
                return
            except Exception as exc:
                logger.warning("OHLCVAggregator[%s] error: %s — retry in %ds",
                               symbol, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _process_bars(self, symbol: str, ohlcv: list) -> None:
        if not ohlcv:
            return
        buf     = self._buffers.setdefault(symbol, CandleBuffer(symbol))
        now_min = int(time.time() // 60) * 60

        for row in ohlcv:
            ts_ms, o, h, l, c, vol = row
            ts_min   = int(ts_ms // 1000 // 60) * 60
            complete = ts_min < now_min
            candle   = Candle(symbol=symbol, ts_open=ts_min,
                              open=float(o), high=float(h),
                              low=float(l),  close=float(c),
                              volume=float(vol), complete=complete)

            if complete and ts_min > self._last_ts.get(symbol, 0):
                buf.push(candle)
                self._last_ts[symbol] = ts_min
                logger.debug("Candle[%s] C=%.4g V=%.4g", symbol, c, vol)
                if self._on_candle:
                    try:
                        self._on_candle(candle)
                    except Exception as exc:
                        logger.warning("on_candle callback error: %s", exc)

    def buffer(self, symbol: str) -> Optional[CandleBuffer]:
        return self._buffers.get(symbol)

    def latest_candle(self, symbol: str) -> Optional[Candle]:
        buf = self._buffers.get(symbol)
        return buf.bars[-1] if buf and buf.bars else None

    def closes(self, symbol: str, n: int = 60) -> list[float]:
        buf = self._buffers.get(symbol)
        return buf.closes(n) if buf else []

    def is_warm(self, symbol: str, min_bars: int = 20) -> bool:
        buf = self._buffers.get(symbol)
        return bool(buf and len(buf) >= min_bars)

    def summary(self) -> str:
        parts = []
        for sym in self._symbols:
            buf = self._buffers.get(sym)
            if buf and buf.bars:
                last = buf.bars[-1]
                parts.append(f"{sym}: {len(buf)} bars  C={last.close:.4g}")
            else:
                parts.append(f"{sym}: warming up")
        return " | ".join(parts)

    @property
    def symbols_warm(self) -> int:
        return sum(1 for s in self._symbols if self.is_warm(s))

    @property
    def total_symbols(self) -> int:
        return len(self._symbols)


_aggregator: Optional[OHLCVAggregator] = None


def get_ohlcv_aggregator(
    symbols:   list[str] | None = None,
    on_candle: Optional[Callable[[Candle], None]] = None,
) -> OHLCVAggregator:
    global _aggregator
    if _aggregator is None:
        _aggregator = OHLCVAggregator(symbols=symbols, on_candle=on_candle)
    return _aggregator


async def start_ohlcv(
    symbols:   list[str] | None = None,
    on_candle: Optional[Callable[[Candle], None]] = None,
) -> OHLCVAggregator:
    agg = get_ohlcv_aggregator(symbols=symbols, on_candle=on_candle)
    await agg.start()
    return agg
