"""DirectDispatcher — routes InboundMessages to DeerTickClient (no HTTP server)."""

from __future__ import annotations

import asyncio
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.channels.message_bus import InboundMessage, InboundMessageType, MessageBus, OutboundMessage
from src.channels.store import ChannelStore

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="deerflow-worker")


class DirectDispatcher:
    """Drop-in replacement for ChannelManager that calls DeerTickClient directly.

    Uses DeerTickClient.stream() (synchronous LangGraph) in a thread executor
    so it never blocks the asyncio event loop that drives the Discord gateway.
    """

    def __init__(self, bus: MessageBus, store: ChannelStore, *, max_concurrency: int = 3) -> None:
        self.bus = bus
        self.store = store
        self._max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None
        self._running = False
        self._task: asyncio.Task | None = None
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from src.client import DeerTickClient
            self._client = DeerTickClient()
            logger.info("DeerTickClient initialized")
        return self._client

    # ── lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._semaphore = asyncio.Semaphore(self._max_concurrency)
        self._task = asyncio.create_task(self._dispatch_loop())
        logger.info("DirectDispatcher started (max_concurrency=%d)", self._max_concurrency)

    async def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("DirectDispatcher stopped")

    # ── dispatch loop ─────────────────────────────────────────────────────────

    async def _dispatch_loop(self) -> None:
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.get_inbound(), timeout=1.0)
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            asyncio.create_task(self._handle_message(msg))

    async def _handle_message(self, msg: InboundMessage) -> None:
        async with self._semaphore:
            try:
                if msg.msg_type == InboundMessageType.COMMAND:
                    await self._handle_command(msg)
                else:
                    await self._handle_chat(msg)
            except Exception:
                logger.exception("Error handling message channel=%s chat=%s", msg.channel_name, msg.chat_id)
                await self._send_reply(msg, "⚠ Internal error — check logs.")

    # ── chat ─────────────────────────────────────────────────────────────────

    async def _handle_chat(self, msg: InboundMessage) -> None:
        # resolve or create thread_id
        thread_id = None
        if msg.topic_id:
            thread_id = self.store.get_thread_id(msg.channel_name, msg.chat_id, topic_id=msg.topic_id)
        if thread_id is None:
            thread_id = str(uuid.uuid4())
            self.store.set_thread_id(
                msg.channel_name, msg.chat_id, thread_id,
                topic_id=msg.topic_id, user_id=msg.user_id,
            )
            logger.info("New thread %s for chat=%s topic=%s", thread_id, msg.chat_id, msg.topic_id)

        # run DeerTickClient.chat() in thread executor — it's synchronous
        loop = asyncio.get_event_loop()
        client = self._get_client()
        try:
            response_text = await loop.run_in_executor(
                _executor,
                lambda: client.chat(msg.text, thread_id=thread_id),
            )
        except Exception as exc:
            logger.exception("DeerTickClient error: %s", exc)
            response_text = f"⚠ Agent error: {exc}"

        if not response_text:
            response_text = "(no response)"

        outbound = OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id=thread_id,
            text=response_text,
            thread_ts=msg.thread_ts,
        )
        await self.bus.publish_outbound(outbound)

    # ── commands ─────────────────────────────────────────────────────────────

    async def _handle_command(self, msg: InboundMessage) -> None:
        cmd = msg.text.strip().lstrip("/").split()[0].lower()
        if cmd == "new":
            new_id = str(uuid.uuid4())
            self.store.set_thread_id(
                msg.channel_name, msg.chat_id, new_id,
                topic_id=msg.topic_id, user_id=msg.user_id,
            )
            reply = f"✓ New conversation started (thread `{new_id[:8]}…`)"
        elif cmd == "status":
            tid = self.store.get_thread_id(msg.channel_name, msg.chat_id, topic_id=msg.topic_id)
            reply = f"Active thread: `{tid}`" if tid else "No active conversation."
        elif cmd == "help":
            reply = (
                "**DeerFlow commands**\n"
                "`/new` — start a fresh conversation\n"
                "`/status` — show current thread ID\n"
                "`/help` — this message"
            )
        else:
            reply = f"Unknown command `/{cmd}`. Try `/help`."
        await self._send_reply(msg, reply)

    async def _send_reply(self, msg: InboundMessage, text: str) -> None:
        tid = self.store.get_thread_id(msg.channel_name, msg.chat_id) or ""
        await self.bus.publish_outbound(OutboundMessage(
            channel_name=msg.channel_name,
            chat_id=msg.chat_id,
            thread_id=tid,
            text=text,
            thread_ts=msg.thread_ts,
        ))
