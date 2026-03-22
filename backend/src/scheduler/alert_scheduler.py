"""alert_scheduler.py — rebalance / threshold / arb loops routed through trading DAG.

Three concurrent loops:
  1. rebalance_loop   every REBALANCE_INTERVAL_SEC (default 300s)
  2. threshold_loop   every THRESHOLD_POLL_SEC      (default 60s)
  3. arb_loop         every ARB_SCAN_INTERVAL_SEC   (default 60s)

Each loop builds a TradingState trigger dict and invokes the trading DAG
(analysts → debate → planner). The formatted recommendation is published
to Discord. If verdict=APPROVE the Discord message gets ✅/❌ reactions;
reacting ✅ triggers AUTO-tier execution via the discord_channel HITL gate.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.scheduler.quant_memory import MarketSignal, QuantMemory, get_qmem
from src.scheduler.quant_memory_v2 import MarketSignalV2, get_qmem_v2

logger = logging.getLogger(__name__)

# ── tunables ──────────────────────────────────────────────────────────────────
REBALANCE_INTERVAL_SEC = int(os.environ.get("REBALANCE_INTERVAL_SEC", "300"))
THRESHOLD_POLL_SEC     = int(os.environ.get("THRESHOLD_POLL_SEC", "60"))
ARB_SCAN_INTERVAL_SEC  = int(os.environ.get("ARB_SCAN_INTERVAL_SEC", "60"))
VIX_ALERT_THRESHOLD    = float(os.environ.get("VIX_ALERT_THRESHOLD", "25.0"))
VIX_VETO_THRESHOLD     = float(os.environ.get("VIX_VETO_THRESHOLD", "35.0"))
PRICE_MOVE_PCT         = float(os.environ.get("PRICE_MOVE_PCT", "3.0"))
ARB_MIN_PROFIT_PCT     = float(os.environ.get("ARB_MIN_PROFIT_PCT", "0.10"))
MARKET_OPEN_ET_HOUR    = 9
MARKET_OPEN_ET_MIN     = 30
MARKET_CLOSE_ET_HOUR   = 16


@dataclass
class ThresholdState:
    vix_last: float = 0.0
    vix_alerted_at: float = 0.0
    price_baselines: dict[str, float] = field(default_factory=dict)
    price_alerted: dict[str, float] = field(default_factory=dict)


class AlertScheduler:

    def __init__(self, bus, store, target_channel_id: str, client=None) -> None:
        self.bus = bus
        self.store = store
        self.target_channel_id = target_channel_id
        self._client = client
        self._state = ThresholdState()
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._capital_usd: Optional[float] = None

    # ── trading DAG ───────────────────────────────────────────────────────────

    def _get_trading_graph(self):
        from src.agents.trading.graph import get_trading_graph
        return get_trading_graph()

    async def _run_trading_dag(self, trigger: dict[str, Any]) -> str:
        """Invoke the trading DAG with a trigger and return the formatted recommendation."""
        graph = self._get_trading_graph()
        import uuid
        thread_id = f"scheduler-{trigger.get('type','unknown')}-{int(time.time())}"
        initial_state = {
            "messages": [],
            "analyst_reports": [],
            "trigger": trigger,
            "phase": "analysis",
        }
        config = {"configurable": {"thread_id": thread_id}}
        try:
            result = await graph.ainvoke(initial_state, config=config)
            rec = result.get("formatted_recommendation", "")
            # Surface approval_id so discord_channel can attach reactions
            approval_id = result.get("approval_id")
            execution_plan = result.get("execution_plan")
            verdict = result.get("verdict", "HOLD")
            return rec, verdict, approval_id, execution_plan
        except Exception as exc:
            logger.exception("Trading DAG error for trigger=%s", trigger.get("type"))
            return f"Trading DAG error: {exc}", "HOLD", None, None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._rebalance_loop()),
            asyncio.create_task(self._threshold_loop()),
            asyncio.create_task(self._arb_loop()),
        ]
        logger.info(
            "AlertScheduler started — rebalance=%ds threshold=%ds arb=%ds",
            REBALANCE_INTERVAL_SEC, THRESHOLD_POLL_SEC, ARB_SCAN_INTERVAL_SEC,
        )

    async def stop(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass
        logger.info("AlertScheduler stopped")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_rh(self):
        from src.tools.robinhood_tools import _get_session
        return _get_session()

    def _get_capital(self) -> float:
        if self._capital_usd:
            return self._capital_usd
        try:
            rh = self._get_rh()
            profiles = rh.profiles.load_portfolio_profile()
            equity = float(profiles.get("equity") or profiles.get("extended_hours_equity") or 0)
            if equity > 0:
                self._capital_usd = equity
                return equity
        except Exception:
            pass
        return float(os.environ.get("ARB_MAX_TRADE_USD", "50")) * 20

    @staticmethod
    def _market_is_open() -> bool:
        import datetime
        now = datetime.datetime.utcnow()
        et_hour = (now.hour - 4) % 24
        if now.weekday() >= 5:
            return False
        after_open = (et_hour > MARKET_OPEN_ET_HOUR) or (
            et_hour == MARKET_OPEN_ET_HOUR and now.minute >= MARKET_OPEN_ET_MIN
        )
        return after_open and et_hour < MARKET_CLOSE_ET_HOUR

    async def _send(self, text: str, metadata: dict | None = None) -> None:
        from src.channels.message_bus import OutboundMessage
        msg = OutboundMessage(
            channel_name="discord",
            chat_id=self.target_channel_id,
            thread_id="scheduler",
            text=text,
            metadata=metadata or {},
        )
        await self.bus.publish_outbound(msg)

    # ── rebalance loop ────────────────────────────────────────────────────────

    async def _rebalance_loop(self) -> None:
        await asyncio.sleep(10)
        while self._running:
            try:
                await self._run_rebalance()
            except Exception:
                logger.exception("Rebalance loop error")
            await asyncio.sleep(REBALANCE_INTERVAL_SEC)

    async def _run_rebalance(self) -> None:
        import datetime
        logger.info("Running rebalance DAG...")
        await self._send("🔄 **Rebalance scan started…**")
        capital = self._get_capital()
        try:
            qmem_ctx = get_qmem_v2().summary(5)
        except Exception:
            qmem_ctx = get_qmem().summary(5)

        trigger = {
            "type": "rebalance",
            "capital_usd": capital,
            "qmem_ctx": qmem_ctx,
            "vix": self._state.vix_last or 20.0,
            "trade_enabled": os.environ.get("TRADE_ENABLED", "false"),
            "timestamp": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        }

        rec, verdict, approval_id, plan = await self._run_trading_dag(trigger)
        meta = {}
        if verdict == "APPROVE" and approval_id and plan:
            meta = {"approval_id": approval_id, "execution_plan": plan, "needs_reactions": True}

        await self._send(f"📊 **Rebalance Analysis**\n\n{rec}", metadata=meta)

        try:
            get_qmem().push(MarketSignal(vix_fear=self._state.vix_last or 20.0, signal_type=0))
            get_qmem_v2().push(MarketSignalV2(
                vix_fear=self._state.vix_last or 20.0,
                signal_type=0,
                btc_momentum=float(trigger.get('qmem_ctx', '0').split('BTC')[1].split('%')[0])
                if 'BTC' in str(trigger.get('qmem_ctx', '')) else 0.0,
            ))
        except Exception:
            pass

    # ── arb loop ─────────────────────────────────────────────────────────────

    async def _arb_loop(self) -> None:
        await asyncio.sleep(30)
        while self._running:
            try:
                await self._run_arb_scan()
            except Exception:
                logger.exception("Arb loop error")
            await asyncio.sleep(ARB_SCAN_INTERVAL_SEC)

    async def _run_arb_scan(self) -> None:
        from src.arb import arb_tools as _arb_tools
        scanner = _arb_tools._scanner
        if scanner is None:
            return
        if scanner.symbols_ready < scanner.total_symbols:
            return

        opp = scanner.get_latest()
        if opp is None or opp.profit_pct < ARB_MIN_PROFIT_PCT:
            return

        now = time.time()
        # Simple cooldown based on opportunity name to avoid flooding
        cooldown_key = f"_last_arb_{opp.name}"
        last_sent = getattr(self, cooldown_key, 0.0)
        if now - last_sent < ARB_SCAN_INTERVAL_SEC * 0.9:
            return
        setattr(self, cooldown_key, now)

        capital = self._get_capital()
        trigger = {
            "type": "arb",
            "capital_usd": capital,
            "vix": self._state.vix_last or 20.0,
            "trade_enabled": os.environ.get("TRADE_ENABLED", "false"),
            "opportunity": {
                "name": opp.name,
                "profit_pct": opp.profit_pct,
                "legs": opp.legs,
                "gross_profit_usd": opp.gross_profit_usd,
            },
            "age_seconds": round(now - opp.detected_at, 1),
        }

        logger.info("Arb trigger → trading DAG: %s %+.4f%%", opp.name, opp.profit_pct)

        try:
            get_qmem().push(MarketSignal(arb_profit_pct=opp.profit_pct, signal_type=1))
            get_qmem_v2().push(MarketSignalV2(
                arb_profit_pct=opp.profit_pct,
                signal_type=1,
                vix_fear=self._state.vix_last or 20.0,
            ))
        except Exception:
            pass

        rec, verdict, approval_id, plan = await self._run_trading_dag(trigger)
        meta = {}
        if verdict == "APPROVE" and approval_id and plan:
            meta = {"approval_id": approval_id, "execution_plan": plan, "needs_reactions": True}

        await self._send(f"⚡ **Arb Analysis**\n\n{rec}", metadata=meta)

    # ── threshold loop ────────────────────────────────────────────────────────

    async def _threshold_loop(self) -> None:
        await asyncio.sleep(15)
        await self._seed_baselines()
        while self._running:
            try:
                await self._check_vix()
                await self._check_price_moves()
            except Exception:
                logger.exception("Threshold loop error")
            await asyncio.sleep(THRESHOLD_POLL_SEC)

    async def _seed_baselines(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            rh = self._get_rh()
            watchlist = await loop.run_in_executor(
                None, lambda: rh.account.get_watchlist_by_name("Cryptos to Watch")
            )
            symbols = [
                (item if isinstance(item, str) else item.get("symbol", ""))
                for item in (watchlist or [])
            ]
            symbols = [s for s in symbols if s and s.isupper() and s.isalpha() and 1 <= len(s) <= 6]
            if symbols:
                quotes = await loop.run_in_executor(None, lambda: rh.stocks.get_quotes(symbols))
                for q in (quotes or []):
                    if not q:
                        continue
                    sym = q.get("symbol")
                    raw = q.get("last_extended_hours_trade_price") or q.get("last_trade_price")
                    if sym and raw:
                        p = float(raw)
                        if p > 0:
                            self._state.price_baselines[sym] = p
            logger.info("Price baselines seeded: %s", self._state.price_baselines)
        except Exception:
            logger.exception("Failed to seed price baselines")

    async def _check_vix(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            rh = self._get_rh()
            raw = await loop.run_in_executor(None, lambda: rh.stocks.get_latest_price("VIXY"))
            vix = float(raw[0]) if raw and raw[0] else 0.0
            if vix == 0.0:
                return
            now = time.time()
            cooldown = 1800
            _vix_path = pathlib.Path(".deer-flow/vix_alerted_at.txt")
            try:
                self._state.vix_alerted_at = float(_vix_path.read_text())
            except Exception:
                pass

            if vix >= VIX_VETO_THRESHOLD:
                if now - self._state.vix_alerted_at > cooldown:
                    await self._send(
                        f"🚨 **EXTREME FEAR** — VIXY ${vix:.2f} ≥ veto threshold ${VIX_VETO_THRESHOLD}\n"
                        "All buy orders vetoed. Defensive positioning recommended."
                    )
                    self._state.vix_alerted_at = now
                    _vix_path.parent.mkdir(exist_ok=True)
                    _vix_path.write_text(str(now))
            elif vix >= VIX_ALERT_THRESHOLD:
                if now - self._state.vix_alerted_at > cooldown:
                    await self._send(
                        f"⚠️ **ELEVATED FEAR** — VIXY ${vix:.2f} ≥ caution threshold ${VIX_ALERT_THRESHOLD}"
                    )
                    self._state.vix_alerted_at = now
                    _vix_path.parent.mkdir(exist_ok=True)
                    _vix_path.write_text(str(now))
            self._state.vix_last = vix
        except Exception:
            logger.exception("VIX check failed")

    async def _check_price_moves(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            rh = self._get_rh()
            baselines = list(self._state.price_baselines.items())
            symbols = [s for s, b in baselines if b > 0.0]
            if not symbols:
                return
            quotes = await loop.run_in_executor(None, lambda: rh.stocks.get_quotes(symbols))
            for q in (quotes or []):
                if not q:
                    continue
                sym = q.get("symbol")
                raw = q.get("last_extended_hours_trade_price") or q.get("last_trade_price")
                baseline = self._state.price_baselines.get(sym)
                if not baseline or not raw:
                    continue
                price = float(raw)
                if price == 0.0:
                    continue
                pct_move = ((price - baseline) / baseline) * 100
                now = time.time()
                last_alerted = self._state.price_alerted.get(sym, 0.0)
                if abs(pct_move) >= PRICE_MOVE_PCT and now - last_alerted > 900:
                    direction = "📈" if pct_move > 0 else "📉"
                    await self._send(
                        f"{direction} **Price Move: {sym}** "
                        f"${baseline:.2f} → ${price:.2f} ({pct_move:+.1f}%)"
                    )
                    self._state.price_alerted[sym] = now
                    self._state.price_baselines[sym] = price

                    # Also route threshold triggers through the DAG for significant moves
                    if abs(pct_move) >= PRICE_MOVE_PCT * 2:
                        trigger = {
                            "type": "threshold",
                            "symbol": sym,
                            "pct_move": pct_move,
                            "current_price": price,
                            "baseline_price": baseline,
                            "capital_usd": self._get_capital(),
                            "vix": self._state.vix_last or 20.0,
                        }
                        rec, verdict, approval_id, plan = await self._run_trading_dag(trigger)
                        meta = {}
                        if verdict == "APPROVE" and approval_id and plan:
                            meta = {"approval_id": approval_id, "execution_plan": plan, "needs_reactions": True}
                        await self._send(f"📊 **Threshold Analysis: {sym}**\n\n{rec}", metadata=meta)
        except Exception:
            logger.exception("Price move check failed")
