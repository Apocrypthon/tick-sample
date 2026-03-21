"""
Quantized scheduler memory — 8^4 = 4096-bit (512-byte) ring buffer.

Each market signal is encoded as 32 bytes (8 fields × 4 bits each):
  Field 0: BTC momentum   [0=crash, 7=flat, 15=moon]
  Field 1: ETH momentum   [same scale]
  Field 2: SOL momentum   [same scale]
  Field 3: VIX fear       [0=calm, 15=extreme]
  Field 4: SPY trend      [0=crash, 7=flat, 15=bull]
  Field 5: arb_profit_pct [0=none, 15=>=0.30%]
  Field 6: portfolio_pct  [0=0%, 15=100% deployed]
  Field 7: signal_type    [0=rebalance,1=arb,2=threshold,3=manual]

Ring holds 16 signals = 16 × 32 = 512 bytes = 4096 bits = 8^4 bits.
"""
from __future__ import annotations
import json, math, os, pathlib, time
from dataclasses import dataclass, field
from typing import Optional

RING_SIZE   = 16   # number of signals stored
FIELD_COUNT = 8    # fields per signal
BITS        = 4    # bits per field
LEVELS      = 2**BITS  # 16 quantization levels (0-15)

_PATH = pathlib.Path(os.path.expanduser(
    "~/.deer-tick-qmem.json"
))

SIGNAL_TYPES = {0: "rebalance", 1: "arb", 2: "threshold", 3: "manual"}


def _quant(value: float, lo: float, hi: float) -> int:
    """Quantize float in [lo,hi] to [0, LEVELS-1]."""
    if hi == lo:
        return 0
    clamped = max(lo, min(hi, value))
    return round((clamped - lo) / (hi - lo) * (LEVELS - 1))


def _dequant(q: int, lo: float, hi: float) -> float:
    return lo + (q / (LEVELS - 1)) * (hi - lo)


@dataclass
class MarketSignal:
    btc_momentum:   float = 0.0   # price % change from baseline, [-5, +5]
    eth_momentum:   float = 0.0
    sol_momentum:   float = 0.0
    vix_fear:       float = 20.0  # VIXY price [15, 50]
    spy_trend:      float = 0.0   # SPY % from 20d ma [-5, +5]
    arb_profit_pct: float = 0.0   # [0, 0.50]
    portfolio_pct:  float = 0.0   # % deployed [0, 1.0]
    signal_type:    int   = 0     # 0-3

    timestamp: float = field(default_factory=time.time)

    def encode(self) -> list[int]:
        return [
            _quant(self.btc_momentum,   -5.0,  5.0),
            _quant(self.eth_momentum,   -5.0,  5.0),
            _quant(self.sol_momentum,   -5.0,  5.0),
            _quant(self.vix_fear,       15.0, 50.0),
            _quant(self.spy_trend,      -5.0,  5.0),
            _quant(self.arb_profit_pct,  0.0,  0.5),
            _quant(self.portfolio_pct,   0.0,  1.0),
            min(3, max(0, self.signal_type)),
        ]

    @classmethod
    def decode(cls, q: list[int], ts: float = 0.0) -> "MarketSignal":
        return cls(
            btc_momentum   = _dequant(q[0], -5.0,  5.0),
            eth_momentum   = _dequant(q[1], -5.0,  5.0),
            sol_momentum   = _dequant(q[2], -5.0,  5.0),
            vix_fear       = _dequant(q[3], 15.0, 50.0),
            spy_trend      = _dequant(q[4], -5.0,  5.0),
            arb_profit_pct = _dequant(q[5],  0.0,  0.5),
            portfolio_pct  = _dequant(q[6],  0.0,  1.0),
            signal_type    = q[7],
            timestamp      = ts,
        )

    def summary(self) -> str:
        stype = SIGNAL_TYPES.get(self.signal_type, "?")
        import datetime
        ts = datetime.datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return (
            f"[{ts}|{stype}] "
            f"BTC{self.btc_momentum:+.1f}% ETH{self.eth_momentum:+.1f}% "
            f"SOL{self.sol_momentum:+.1f}% VIX={self.vix_fear:.1f} "
            f"arb={self.arb_profit_pct:.3f}% deployed={self.portfolio_pct:.0%}"
        )


class QuantMemory:
    """512-byte ring buffer for scheduler market signals."""

    def __init__(self):
        self._ring: list[dict] = []
        self._load()

    def _load(self):
        try:
            if _PATH.exists():
                data = json.loads(_PATH.read_text())
                self._ring = data.get("ring", [])[-RING_SIZE:]
        except Exception:
            self._ring = []

    def _save(self):
        try:
            _PATH.write_text(json.dumps({"ring": self._ring[-RING_SIZE:]}, indent=None))
        except Exception:
            pass

    def push(self, signal: MarketSignal):
        self._ring.append({"q": signal.encode(), "ts": signal.timestamp})
        if len(self._ring) > RING_SIZE:
            self._ring = self._ring[-RING_SIZE:]
        self._save()

    def last(self, n: int = 1) -> list[MarketSignal]:
        entries = self._ring[-n:]
        return [MarketSignal.decode(e["q"], e.get("ts", 0)) for e in entries]

    def summary(self, n: int = 5) -> str:
        signals = self.last(n)
        if not signals:
            return "No quantized memory yet."
        lines = [f"QMem last {len(signals)} signals (8^4 ring):"]
        lines += [f"  {s.summary()}" for s in signals]
        return "\n".join(lines)

    @property
    def bits_used(self) -> int:
        return len(self._ring) * FIELD_COUNT * BITS

    @property
    def bits_total(self) -> int:
        return RING_SIZE * FIELD_COUNT * BITS  # 4096


# Singleton
_qmem: Optional[QuantMemory] = None

def get_qmem() -> QuantMemory:
    global _qmem
    if _qmem is None:
        _qmem = QuantMemory()
    return _qmem
