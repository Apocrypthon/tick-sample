"""
HITL approval gate + dry-run Discord thread manager.

Dry-run thread behaviour (Conflict HITL-B resolution):
  - Thread created in DISCORD_ALERT_CHANNEL_ID
  - Name: 🔴 dry-run · {DEERTICK_MODEL} · {YYYY-MM-DD}
  - Each dry-run order posted as a formatted message inside the thread
  - Hourly: thread archived, new one opened — keeps alert channel clean
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ApprovalResult:
    approval_id:    str
    approved:       bool
    execution_plan: Optional[dict] = None


class HITLGate:
    """asyncio.Lock + per-approval asyncio.Event. Thread-safe."""

    def __init__(self, timeout_seconds: int = 600):
        self._timeout  = timeout_seconds
        self._lock     = asyncio.Lock()
        self._pending: dict[str, asyncio.Event]  = {}
        self._results: dict[str, ApprovalResult] = {}

    async def wait_for_approval(self, approval_id: str) -> ApprovalResult:
        event = asyncio.Event()
        async with self._lock:
            self._pending[approval_id] = event
        try:
            await asyncio.wait_for(event.wait(), timeout=self._timeout)
        except asyncio.TimeoutError:
            logger.warning("HITL timeout: %s", approval_id)
            return ApprovalResult(approval_id=approval_id, approved=False)
        finally:
            async with self._lock:
                self._pending.pop(approval_id, None)
        return self._results.pop(approval_id, ApprovalResult(approval_id, False))

    async def resolve(self, approval_id: str, approved: bool,
                      plan: Optional[dict] = None) -> None:
        result = ApprovalResult(approval_id, approved, plan)
        async with self._lock:
            self._results[approval_id] = result
            event = self._pending.get(approval_id)
        if event:
            event.set()


class DryRunThreadManager:
    """
    Rotating Discord thread for dry-run order visibility.

    One active thread per hour. Updated as orders arrive.
    Archived + reopened on the hour boundary.
    Thread name includes DEERTICK_MODEL and date from env.
    """

    def __init__(self, client: Any, channel_id: int):
        self._client     = client
        self._channel_id = channel_id
        self._thread     = None
        self._model      = os.getenv("DEERTICK_MODEL", "claude-sonnet-4-6")
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        await self._ensure_thread()
        self._task = asyncio.create_task(self._hourly_cycle())
        logger.info("DryRunThreadManager started (model=%s)", self._model)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def post_order(self, plan: dict, reason: str) -> None:
        await self._ensure_thread()
        if not self._thread:
            return
        trade_enabled = os.getenv("TRADE_ENABLED", "false")
        msg = (
            f"**🔴 DRY RUN** — `TRADE_ENABLED={trade_enabled}`\n"
            f"```\n"
            f"Symbol   : {plan.get('symbol', 'N/A')}\n"
            f"Side     : {plan.get('side', 'N/A')}\n"
            f"Notional : ${plan.get('notional_usd', 0.0):,.2f}\n"
            f"AUTO tier: ${plan.get('copy_trade_auto', 0.0):,.2f}\n"
            f"Reason   : {reason}\n"
            f"Model    : {self._model}\n"
            f"UTC      : {datetime.datetime.utcnow().strftime('%H:%M:%S')}\n"
            f"```\n"
            f"*Set `TRADE_ENABLED=true` in `.env` + restart to go live.*"
        )
        try:
            await self._thread.send(msg)
        except Exception as exc:
            logger.warning("DryRunThread.send failed: %s", exc)
            self._thread = None

    async def _ensure_thread(self) -> None:
        if self._thread:
            return
        try:
            channel = self._client.get_channel(self._channel_id)
            if channel is None:
                channel = await self._client.fetch_channel(self._channel_id)
            name = self._thread_name()
            # Reuse archived thread if same name exists
            async for t in channel.archived_threads(limit=10):
                if t.name == name:
                    await t.edit(archived=False)
                    self._thread = t
                    logger.info("Reopened dry-run thread: %s", name)
                    return
            self._thread = await channel.create_thread(
                name=name,
                auto_archive_duration=60,
            )
            await self._thread.send(
                f"## 🔴 Dry-Run Monitor\n"
                f"**Model:** `{self._model}`  "
                f"**Date:** `{datetime.date.today()}`\n"
                f"Each dry-run order will appear here. "
                f"Thread rotates hourly and is archived automatically.\n"
                f"→ Set `TRADE_ENABLED=true` in `.env` to go live."
            )
            logger.info("Created dry-run thread: %s", name)
        except Exception as exc:
            logger.error("Could not create dry-run thread: %s", exc)

    def _thread_name(self) -> str:
        return f"🔴 dry-run · {self._model} · {datetime.date.today()}"

    async def _hourly_cycle(self) -> None:
        while True:
            await asyncio.sleep(3600)
            try:
                if self._thread:
                    await self._thread.send(
                        "⏰ *Hourly rotation — archiving this thread.*"
                    )
                    await self._thread.edit(archived=True, locked=True)
                    self._thread = None
                    logger.info("Dry-run thread archived, will reopen on next order")
                await self._ensure_thread()
            except Exception as exc:
                logger.warning("Hourly dry-run rotation error: %s", exc)
                self._thread = None


_gate: Optional[HITLGate] = None

def get_hitl_gate() -> HITLGate:
    global _gate
    if _gate is None:
        _gate = HITLGate()
    return _gate
