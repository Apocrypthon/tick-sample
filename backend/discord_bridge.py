"""discord_bridge.py — single entry point for the DeerTick trading agent."""

import asyncio
import logging
import os
import signal
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s -- %(message)s",
)
logger = logging.getLogger("discord_bridge")


async def main() -> None:
    from src.channels.direct_dispatcher import DirectDispatcher
    from src.channels.discord_channel import DiscordChannel
    from src.hitl.gate import get_hitl_gate, DryRunThreadManager
    from src.channels.message_bus import MessageBus

    from src.channels.store import ChannelStore
    from src.config.app_config import get_app_config
    import src.tools as _tools_pkg
    from src.tools.robinhood_tools import ROBINHOOD_TOOLS
    from src.tools.strategy_tools import STRATEGY_TOOLS
    from src.tools.coinbase_tools import COINBASE_TOOLS
    from src.tools.alpaca_tools import ALPACA_TOOLS
    from src.arb.arb_tools import ARB_TOOLS, set_scanner
    from src.arb.arb_scanner import ArbScanner
    from src.scheduler.alert_scheduler import AlertScheduler
    from src.webhook_server import app as webhook_app, init_webhook

    # -- config ----------------------------------------------------------------
    config = get_app_config()
    extra = config.model_extra or {}
    channels_cfg = extra.get("channels", {})
    discord_cfg = dict(channels_cfg.get("discord", {}))
    discord_cfg["bot_token"] = os.environ["DISCORD_BOT_TOKEN"]

    if not discord_cfg.get("enabled", False):
        logger.error("Discord not enabled in config.yaml")
        return

    # -- arb scanner -----------------------------------------------------------
    scanner = ArbScanner()
    await scanner.start()

    # DATA-001: start live 1m candle aggregator
    from src.data.ws_ohlcv import start_ohlcv
    ohlcv = await start_ohlcv(
        symbols=["BTC/USD", "ETH/USD", "SOL/USD"]
    )
    logger.info("OHLCVAggregator started (%d symbols)", ohlcv.total_symbols)
    set_scanner(scanner)
    logger.info("ArbScanner started (%d symbols)", scanner.total_symbols)

    # -- patch tool registry (patch package __init__ so all importers see it) --
    all_extra = ROBINHOOD_TOOLS + STRATEGY_TOOLS + COINBASE_TOOLS + ALPACA_TOOLS + ARB_TOOLS
    _original = _tools_pkg.get_available_tools

    def _patched(**kwargs):
        return _original(**kwargs) + all_extra

    # Patch both the package and the submodule so every import path is covered
    _tools_pkg.get_available_tools = _patched
    if hasattr(_tools_pkg, "tools") and hasattr(_tools_pkg.tools, "get_available_tools"):
        _tools_pkg.tools.get_available_tools = _patched
    logger.info("Tools registered (%d): %s", len(all_extra), [t.name for t in all_extra])

    # -- wire components -------------------------------------------------------
    bus = MessageBus()
    store = ChannelStore()
    dispatcher = DirectDispatcher(bus=bus, store=store)
    channel = DiscordChannel(bus=bus, config=discord_cfg)
    # Attach HITL gate singleton — used by execution node to await approvals
    hitl_gate = get_hitl_gate()
    logger.info("HITLGate initialized (timeout=%ds)", hitl_gate._timeout)
    target_channel_id = os.environ.get("DISCORD_ALERT_CHANNEL_ID", "")

    # Patch memory middleware to skip scheduler threads (prevents poisoned facts)
    try:
        import src.config.memory_config as _mc
        _orig_mem_enabled = _mc.get_memory_config
        def _patched_mem_config():
            cfg = _orig_mem_enabled()
            return cfg
        # Skip memory injection for scheduler-* thread IDs
        import src.agents.middlewares.memory_middleware as _mm
        _orig_should_update = _mm.MemoryMiddleware._should_update_memory if hasattr(_mm, "MemoryMiddleware") else None
        if _orig_should_update:
            def _skip_scheduler(self, state, **kw):
                thread_id = state.get("thread_id", "") or ""
                if str(thread_id).startswith("scheduler") or str(thread_id).startswith("arb-"):
                    return False
                return _orig_should_update(self, state, **kw)
            _mm.MemoryMiddleware._should_update_memory = _skip_scheduler
            logger.info("Memory middleware: scheduler threads excluded")
    except Exception as _e:
        logger.debug("Memory skip patch skipped: %s", _e)

    scheduler = AlertScheduler(bus=bus, store=store, target_channel_id=target_channel_id)

    # -- webhook server --------------------------------------------------------
    loop = asyncio.get_event_loop()
    init_webhook(bus, loop)

    def _run_webhook():
        import uvicorn
        uvicorn.run(webhook_app, host="0.0.0.0", port=8080, log_level="warning")

    webhook_thread = threading.Thread(target=_run_webhook, daemon=True)
    webhook_thread.start()
    logger.info("Webhook server started on :8080")

    # -- pyngrok tunnel --------------------------------------------------------
    ngrok_token = os.environ.get("NGROK_AUTHTOKEN", "")
    if ngrok_token:
        try:
            from pyngrok import ngrok as _ngrok, conf as _conf
            _conf.get_default().auth_token = ngrok_token
            # Kill any stale tunnel holding the static domain before rebinding
            try:
                _ngrok.kill()
            except Exception:
                pass
            await asyncio.sleep(1)
            tunnel = _ngrok.connect(
                addr=8080,
                proto="http",
                domain=os.environ.get("NGROK_DOMAIN", ""),
            )
            public_url = tunnel.public_url
            logger.info("ngrok tunnel: %s", public_url)
            logger.info("TradingView webhook URL: %s/webhook", public_url)
        except Exception as exc:
            logger.warning("pyngrok failed (non-fatal): %s", exc)
    else:
        logger.warning("NGROK_AUTHTOKEN not set -- tunnel skipped. Run: ngrok http 8080")

    # -- start -----------------------------------------------------------------
    await dispatcher.start()
    await channel.start()
    if target_channel_id:
        await scheduler.start()
        logger.info("AlertScheduler active -- channel %s", target_channel_id)
    logger.info("DeerTick agent running -- Ctrl+C to stop")

    # -- run until signal ------------------------------------------------------
    stop_event = asyncio.Event()

    def _handle_signal(*_):
        logger.info("Shutdown signal received")
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _handle_signal)

    await stop_event.wait()

    # -- shutdown --------------------------------------------------------------
    logger.info("Shutting down...")
    await scanner.stop()
    await channel.stop()
    await dispatcher.stop()
    if target_channel_id:
        await scheduler.stop()
    logger.info("Clean exit.")


if __name__ == "__main__":
    asyncio.run(main())
