"""Discord channel — gateway bot with HITL approval gate via message reactions."""
from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from typing import Any

from src.channels.base import Channel
from src.channels.message_bus import (
    InboundMessage,
    InboundMessageType,
    MessageBus,
    OutboundMessage,
    ResolvedAttachment,
)

logger = logging.getLogger(__name__)

_DISCORD_MAX_CHARS = 1900
_APPROVE_EMOJI = "✅"
_REJECT_EMOJI  = "❌"


def _chunk_text(text: str) -> list[str]:
    if len(text) <= _DISCORD_MAX_CHARS:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:_DISCORD_MAX_CHARS])
        text = text[_DISCORD_MAX_CHARS:]
    return chunks


class DiscordChannel(Channel):
    """Discord bot channel with HITL approval gate.

    Approval flow:
      1. Scheduler publishes OutboundMessage with metadata={"needs_reactions": True,
         "approval_id": str, "execution_plan": dict}.
      2. This channel sends the message, adds ✅/❌ reactions, and stores
         {discord_message_id: {approval_id, execution_plan, chat_id}} in
         _pending_approvals.
      3. When allowed_user reacts ✅  → publishes InboundMessage:
         "__EXECUTE__ <approval_id> <execution_plan_json>"
      4. When allowed_user reacts ❌  → publishes InboundMessage:
         "__REJECT__ <approval_id>"
      5. The lead agent handles __EXECUTE__ / __REJECT__ commands and calls
         execution tools (or no-ops) accordingly.
    """

    def __init__(self, bus: MessageBus, config: dict[str, Any]) -> None:
        super().__init__(name="discord", bus=bus, config=config)
        self._client = None
        self._thread: threading.Thread | None = None
        self._discord_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._allowed_users: set[int] = set()
        for uid in config.get("allowed_users", []):
            try:
                self._allowed_users.add(int(uid))
            except (ValueError, TypeError):
                pass
        # Maps discord_message_id (str) → {approval_id, execution_plan, chat_id}
        self._pending_approvals: dict[str, dict] = {}

    async def start(self) -> None:
        if self._running:
            return
        try:
            import discord
        except ImportError:
            logger.error("discord.py not installed. Run: uv add discord.py")
            return

        bot_token = self.config.get("bot_token", "")
        if not bot_token:
            logger.error("Discord channel requires bot_token in config")
            return

        self._main_loop = asyncio.get_event_loop()
        self._running = True
        self.bus.subscribe_outbound(self._on_outbound)

        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        client = discord.Client(intents=intents)
        self._client = client

        @client.event
        async def on_ready():
            logger.info("Discord bot: %s (id=%s)", client.user, client.user.id)

        @client.event
        async def on_message(message):
            if message.author == client.user:
                return
            if not self._check_user(message.author.id):
                return
            text = message.content.strip()
            if not text:
                return

            chat_id = str(message.channel.id)
            user_id = str(message.author.id)
            msg_id  = str(message.id)

            topic_id = (
                str(message.reference.message_id)
                if message.reference and message.reference.message_id
                else msg_id
            )

            msg_type = (
                InboundMessageType.COMMAND
                if text.startswith("/")
                else InboundMessageType.CHAT
            )

            inbound = self._make_inbound(
                chat_id=chat_id,
                user_id=user_id,
                text=text,
                msg_type=msg_type,
                thread_ts=msg_id,
            )
            inbound.topic_id = topic_id

            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    message.add_reaction("⏳"), self._discord_loop
                )
                asyncio.run_coroutine_threadsafe(
                    self.bus.publish_inbound(inbound), self._main_loop
                )

        @client.event
        async def on_raw_reaction_add(payload):
            """Handle ✅/❌ reactions on pending approval messages."""
            # Ignore bot's own reactions
            if payload.user_id == client.user.id:
                return
            if not self._check_user(payload.user_id):
                return

            msg_id_str = str(payload.message_id)
            if msg_id_str not in self._pending_approvals:
                return

            pending = self._pending_approvals[msg_id_str]
            emoji = str(payload.emoji)

            if emoji == _APPROVE_EMOJI:
                plan_json = json.dumps(pending["execution_plan"])
                text = f"__EXECUTE__ {pending['approval_id']} {plan_json}"
                action = "APPROVED"
            elif emoji == _REJECT_EMOJI:
                text = f"__REJECT__ {pending['approval_id']}"
                action = "REJECTED"
            else:
                return

            # Notify HITLGate so any awaiting coroutine unblocks
            try:
                from src.hitl.gate import get_hitl_gate
                approved = (emoji == _APPROVE_EMOJI)
                plan     = pending.get("execution_plan")
                aid      = pending["approval_id"]
                import asyncio as _aio
                if self._main_loop and self._main_loop.is_running():
                    _aio.run_coroutine_threadsafe(
                        get_hitl_gate().resolve(aid, approved=approved, plan=plan),
                        self._main_loop,
                    )
            except Exception as _ge:
                import logging as _log
                _log.getLogger(__name__).warning("HITLGate.resolve failed: %s", _ge)

            logger.info(
                "HITL gate: %s  approval_id=%s  user=%s",
                action, pending["approval_id"], payload.user_id,
            )

            # Remove from pending so duplicate reactions are ignored
            del self._pending_approvals[msg_id_str]

            inbound = InboundMessage(
                channel_name="discord",
                chat_id=pending["chat_id"],
                user_id=str(payload.user_id),
                text=text,
                msg_type=InboundMessageType.COMMAND,
            )
            if self._main_loop and self._main_loop.is_running():
                asyncio.run_coroutine_threadsafe(
                    self.bus.publish_inbound(inbound), self._main_loop
                )

        self._thread = threading.Thread(
            target=self._run_gateway, args=(bot_token,), daemon=True
        )
        self._thread.start()
        logger.info("Discord channel started")

    async def stop(self) -> None:
        self._running = False
        self.bus.unsubscribe_outbound(self._on_outbound)
        if self._discord_loop and self._client:
            asyncio.run_coroutine_threadsafe(self._client.close(), self._discord_loop)
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None
        logger.info("Discord channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client or not self._discord_loop:
            return

        channel = self._client.get_channel(int(msg.chat_id))
        if channel is None:
            try:
                channel = await asyncio.wrap_future(
                    asyncio.run_coroutine_threadsafe(
                        self._client.fetch_channel(int(msg.chat_id)),
                        self._discord_loop,
                    )
                )
            except Exception:
                logger.error("Discord: cannot resolve channel %s", msg.chat_id)
                return

        sent_message = None
        for chunk in _chunk_text(msg.text):
            fut = asyncio.run_coroutine_threadsafe(channel.send(chunk), self._discord_loop)
            try:
                sent_message = fut.result(timeout=10)
            except Exception:
                logger.exception("Discord: send chunk failed")

        # Attach ✅/❌ reactions and register HITL pending approval
        needs_reactions = msg.metadata.get("needs_reactions", False)
        approval_id = msg.metadata.get("approval_id")
        execution_plan = msg.metadata.get("execution_plan")

        if needs_reactions and approval_id and execution_plan and sent_message:
            discord_msg_id = str(sent_message.id)
            self._pending_approvals[discord_msg_id] = {
                "approval_id": approval_id,
                "execution_plan": execution_plan,
                "chat_id": msg.chat_id,
            }
            for emoji in (_APPROVE_EMOJI, _REJECT_EMOJI):
                asyncio.run_coroutine_threadsafe(
                    sent_message.add_reaction(emoji), self._discord_loop
                )
            logger.info(
                "HITL gate registered: msg_id=%s approval_id=%s",
                discord_msg_id, approval_id,
            )

    async def send_file(self, msg: OutboundMessage, attachment: ResolvedAttachment) -> bool:
        if not self._client or not self._discord_loop:
            return False
        if attachment.size > 8 * 1024 * 1024:
            logger.warning("Discord: file too large (%d bytes), skipping", attachment.size)
            return False
        try:
            import discord
            channel = self._client.get_channel(int(msg.chat_id))
            if channel is None:
                return False
            data = await asyncio.to_thread(lambda: open(attachment.actual_path, "rb").read())
            discord_file = discord.File(io.BytesIO(data), filename=attachment.filename)
            asyncio.run_coroutine_threadsafe(
                channel.send(file=discord_file), self._discord_loop
            )
            return True
        except Exception:
            logger.exception("Discord: send_file failed for %s", attachment.filename)
            return False

    # ── internal ─────────────────────────────────────────────────────────────

    def _check_user(self, user_id: int) -> bool:
        if not self._allowed_users:
            return True
        return user_id in self._allowed_users

    def _run_gateway(self, bot_token: str) -> None:
        self._discord_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._discord_loop)
        try:
            self._discord_loop.run_until_complete(self._client.start(bot_token))
        except Exception:
            if self._running:
                logger.exception("Discord gateway error")
        finally:
            try:
                self._discord_loop.run_until_complete(self._client.close())
            except Exception:
                pass
