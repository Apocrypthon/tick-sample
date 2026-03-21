"""webhook_server.py — FastAPI server that receives TradingView alerts.

TradingView POSTs JSON to /webhook with a secret token.
The signal is routed into the MessageBus as an InboundMessage,
which DirectDispatcher picks up and sends to Claude for evaluation.

Expected TradingView alert JSON payload:
{
    "secret": "your-tv-webhook-secret",
    "symbol": "BTC-USD",
    "side": "BUY",
    "price": 65432.10,
    "indicator": "RSI",
    "timeframe": "5m",
    "message": "RSI crossed below 30 — oversold"
}
"""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

app = FastAPI(title="DeerTick Webhook", docs_url=None, redoc_url=None)

# injected at startup by discord_bridge.py
_bus = None
_main_loop = None


def init_webhook(bus, loop):
    global _bus, _main_loop
    _bus = bus
    _main_loop = loop
    logger.info("Webhook server initialized with MessageBus")


# ── request schema ────────────────────────────────────────────────────────────

class TVSignal(BaseModel):
    secret: str
    symbol: str
    side: str
    price: float = 0.0
    indicator: str = "unknown"
    timeframe: str = "unknown"
    message: str = ""


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "bus_ready": _bus is not None}


@app.post("/webhook")
async def receive_signal(signal: TVSignal, request: Request):
    expected = os.environ.get("TV_WEBHOOK_SECRET", "")
    if not expected:
        raise HTTPException(status_code=500, detail="TV_WEBHOOK_SECRET not configured")
    if signal.secret != expected:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    if _bus is None or _main_loop is None:
        raise HTTPException(status_code=503, detail="MessageBus not ready")

    # ── build natural language prompt for Claude ──────────────────────────────
    side_verb = "buying" if signal.side.upper() == "BUY" else "selling"
    prompt = (
        f"TradingView {signal.indicator} alert on {signal.symbol} ({signal.timeframe}): "
        f"{signal.message}. "
        f"Signal suggests {side_verb}. Current price ~${signal.price:.2f}. "
        f"Run cross_market_context('{signal.symbol}'), then strategy_analyst to evaluate. "
        f"If APPROVE: place_crypto_order and place_alpaca_order simultaneously. "
        f"If VETO: explain why and suggest alternative. "
        f"Post full analysis to Discord."
    )

    from src.channels.message_bus import InboundMessage, InboundMessageType
    import uuid

    msg = InboundMessage(
        channel_name="discord",
        chat_id=os.environ.get("DISCORD_ALERT_CHANNEL_ID", ""),
        user_id="tradingview",
        text=prompt,
        msg_type=InboundMessageType.CHAT,
        topic_id=f"tv-{signal.symbol}-{signal.timeframe}",
    )

    asyncio.run_coroutine_threadsafe(_bus.publish_inbound(msg), _main_loop)
    logger.info("TradingView signal routed: %s %s @ $%.2f", signal.side, signal.symbol, signal.price)

    return {
        "status": "routed",
        "symbol": signal.symbol,
        "side": signal.side,
        "price": signal.price,
    }
