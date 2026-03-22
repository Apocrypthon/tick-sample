"""
quant_memory_v2 — 16-field × 4-bit × 16-signal ring = 1024 bits.
Sits alongside v1; import from src.scheduler.quant_memory_v2.
"""
from __future__ import annotations
import json, os, pathlib, time
from dataclasses import dataclass, field

RING_SIZE   = 16
FIELD_COUNT = 16
BITS        = 4
LEVELS      = 2 ** BITS

_PATH = pathlib.Path(os.path.expanduser("~/.deer-tick-qmem-v2.json"))
SIGNAL_TYPES = {0: "rebalance", 1: "arb", 2: "threshold", 3: "manual"}
_REGIMES     = {0: "SIDEWAYS", 1: "BULL", 2: "BEAR", 3: "VOLATILE"}


def _q(v: float, lo: float, hi: float) -> int:
    if hi == lo:
        return 0
    return round(max(0, min(LEVELS-1, (max(lo, min(hi, v)) - lo) / (hi - lo) * (LEVELS-1))))


def _dq(q: int, lo: float, hi: float) -> float:
    return lo + (q / (LEVELS - 1)) * (hi - lo)


@dataclass
class MarketSignalV2:
    # v1 fields 0-7
    btc_momentum:   float = 0.0
    eth_momentum:   float = 0.0
    sol_momentum:   float = 0.0
    vix_fear:       float = 20.0
    spy_trend:      float = 0.0
    arb_profit_pct: float = 0.0
    portfolio_pct:  float = 0.0
    signal_type:    int   = 0
    # v2 fields 8-15
    signal_score:   float = 0.0
    rsi_norm:       float = 0.5
    regime_idx:     int   = 0
    vol_ratio:      float = 1.0
    fear_greed:     int   = 50
    corr_btc_eth:   float = 0.5
    spread_bps:     float = 0.0
    kelly_fraction: float = 0.0
    timestamp:      float = field(default_factory=time.time)

    def encode(self) -> list[int]:
        return [
            _q(self.btc_momentum,   -5.0,  5.0),
            _q(self.eth_momentum,   -5.0,  5.0),
            _q(self.sol_momentum,   -5.0,  5.0),
            _q(self.vix_fear,       15.0, 50.0),
            _q(self.spy_trend,      -5.0,  5.0),
            _q(self.arb_profit_pct,  0.0,  0.5),
            _q(self.portfolio_pct,   0.0,  1.0),
            min(3, max(0, self.signal_type)),
            _q(self.signal_score,   -1.0,  1.0),
            _q(self.rsi_norm,        0.0,  1.0),
            min(3, max(0, self.regime_idx)),
            _q(self.vol_ratio,       0.0,  3.0),
            _q(float(self.fear_greed), 0.0, 100.0),
            _q(self.corr_btc_eth,   -1.0,  1.0),
            _q(self.spread_bps,      0.0, 100.0),
            _q(self.kelly_fraction,  0.0,  0.25),
        ]

    @classmethod
    def decode(cls, q: list[int], ts: float = 0.0) -> MarketSignalV2:
        return cls(
            btc_momentum   = _dq(q[0], -5.0,  5.0),
            eth_momentum   = _dq(q[1], -5.0,  5.0),
            sol_momentum   = _dq(q[2], -5.0,  5.0),
            vix_fear       = _dq(q[3], 15.0, 50.0),
            spy_trend      = _dq(q[4], -5.0,  5.0),
            arb_profit_pct = _dq(q[5],  0.0,  0.5),
            portfolio_pct  = _dq(q[6],  0.0,  1.0),
            signal_type    = q[7],
            signal_score   = _dq(q[8], -1.0,  1.0),
            rsi_norm       = _dq(q[9],  0.0,  1.0),
            regime_idx     = q[10],
            vol_ratio      = _dq(q[11], 0.0,  3.0),
            fear_greed     = int(_dq(q[12], 0.0, 100.0)),
            corr_btc_eth   = _dq(q[13], -1.0,  1.0),
            spread_bps     = _dq(q[14],  0.0, 100.0),
            kelly_fraction = _dq(q[15],  0.0,  0.25),
            timestamp      = ts,
        )

    def summary(self) -> str:
        import datetime
        t = datetime.datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S")
        return (f"[{t}|{SIGNAL_TYPES.get(self.signal_type,'?')}] "
                f"BTC{self.btc_momentum:+.1f}% VIX={self.vix_fear:.1f} "
                f"score={self.signal_score:+.2f} RSI={self.rsi_norm*100:.0f} "
                f"F/G={self.fear_greed} regime={_REGIMES[self.regime_idx]} "
                f"kelly={self.kelly_fraction:.1%}")


class QuantMemoryV2:
    def __init__(self):
        self._ring: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            if _PATH.exists():
                self._ring = json.loads(_PATH.read_text()).get("ring", [])[-RING_SIZE:]
        except Exception:
            self._ring = []

    def _save(self) -> None:
        try:
            _PATH.write_text(json.dumps({"ring": self._ring[-RING_SIZE:], "v": 2}))
        except Exception:
            pass

    def push(self, sig: MarketSignalV2) -> None:
        self._ring.append({"q": sig.encode(), "ts": sig.timestamp})
        if len(self._ring) > RING_SIZE:
            self._ring = self._ring[-RING_SIZE:]
        self._save()

    def last(self, n: int = 1) -> list[MarketSignalV2]:
        return [MarketSignalV2.decode(e["q"], e.get("ts", 0.0)) for e in self._ring[-n:]]

    def summary(self, n: int = 5) -> str:
        sigs = self.last(n)
        if not sigs:
            return "No quantized memory yet."
        lines = [f"QMemV2 last {len(sigs)} signals ({RING_SIZE*FIELD_COUNT*BITS} bits):"]
        lines += [f"  {s.summary()}" for s in sigs]
        return "\n".join(lines)

    @property
    def bits_used(self) -> int:
        return len(self._ring) * FIELD_COUNT * BITS

    @property
    def bits_total(self) -> int:
        return RING_SIZE * FIELD_COUNT * BITS


_instance: QuantMemoryV2 | None = None

def get_qmem_v2() -> QuantMemoryV2:
    global _instance
    if _instance is None:
        _instance = QuantMemoryV2()
    return _instance
