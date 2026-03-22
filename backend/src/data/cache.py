"""
cache.py — TTL key-value cache for market data (quotes, OHLCV snapshots).
No external deps — uses stdlib threading + time.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class _Entry:
    value:      Any
    expires_at: float


class TTLCache:
    """Thread-safe TTL cache with optional max-size eviction (LRU)."""

    def __init__(self, maxsize: int = 512, ttl: float = 60.0):
        self._maxsize = maxsize
        self._ttl     = ttl
        self._store:  dict[str, _Entry] = {}
        self._order:  list[str]         = []   # insertion order for LRU
        self._lock    = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                self._evict(key)
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expires = time.monotonic() + (ttl if ttl is not None else self._ttl)
        with self._lock:
            if key in self._store:
                self._order.remove(key)
            elif len(self._store) >= self._maxsize:
                # evict oldest
                oldest = self._order[0]
                self._evict(oldest)
            self._store[key] = _Entry(value, expires)
            self._order.append(key)

    def delete(self, key: str) -> None:
        with self._lock:
            self._evict(key)

    def _evict(self, key: str) -> None:
        self._store.pop(key, None)
        try:
            self._order.remove(key)
        except ValueError:
            pass

    def purge_expired(self) -> int:
        now = time.monotonic()
        with self._lock:
            expired = [k for k, e in self._store.items() if now > e.expires_at]
            for k in expired:
                self._evict(k)
        return len(expired)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# ── domain-specific caches ────────────────────────────────────────────────

# Quote cache: 10s TTL — bid/ask refresh at ~2s from Coinbase WS
_quote_cache   = TTLCache(maxsize=128, ttl=10.0)

# OHLCV snapshot cache: 65s TTL — one full minute plus a small buffer
_ohlcv_cache   = TTLCache(maxsize=256, ttl=65.0)

# Portfolio cache: 30s TTL — avoid hammering broker REST
_portfolio_cache = TTLCache(maxsize=8, ttl=30.0)


def cache_quote(symbol: str, data: dict) -> None:
    _quote_cache.set(f"quote:{symbol}", data)

def get_cached_quote(symbol: str) -> Optional[dict]:
    return _quote_cache.get(f"quote:{symbol}")

def cache_ohlcv(symbol: str, bars: list) -> None:
    _ohlcv_cache.set(f"ohlcv:{symbol}", bars)

def get_cached_ohlcv(symbol: str) -> Optional[list]:
    return _ohlcv_cache.get(f"ohlcv:{symbol}")

def cache_portfolio(venue: str, data: str) -> None:
    _portfolio_cache.set(f"portfolio:{venue}", data)

def get_cached_portfolio(venue: str) -> Optional[str]:
    return _portfolio_cache.get(f"portfolio:{venue}")
