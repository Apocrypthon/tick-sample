# DeerTick

Autonomous trading agent built on LangGraph + Discord. Fork of ByteDance DeerFlow.

## Quick start
```bash
cd ~/deer-tick
./boot.sh                              # full pre-flight + launch
./boot.sh --model=claude-opus-4-6     # override model for this session
./boot.sh --dry-run                   # check keys without launching
```

## Architecture
```
TradingView alert
  → ngrok HTTPS (rene-aristolochiaceous-ayanna.ngrok-free.dev)
  → FastAPI webhook_server.py (:8080)
  → MessageBus → DirectDispatcher
  → DeerFlowClient (claude-sonnet-4-6 via LangGraph)
  → strategy_analyst veto layer
  → place_crypto_order (Coinbase live) + place_alpaca_order (paper parallel)
  → Discord alert
```

## Tools (13 registered)

| Tool | Source | Gate |
|------|--------|------|
| get_portfolio | Robinhood | none |
| get_quote | Robinhood | none |
| get_watchlist | Robinhood | none |
| place_order | Robinhood equity | TRADE_ENABLED |
| cross_market_context | Robinhood + Coinbase | none |
| strategy_analyst | Robinhood | TRADE_ENABLED check |
| get_crypto_quote | Coinbase | none |
| get_crypto_portfolio | Coinbase | none |
| place_crypto_order | Coinbase live | TRADE_ENABLED |
| get_alpaca_portfolio | Alpaca paper | none |
| place_alpaca_order | Alpaca paper | none (always paper) |
| arb_scan | ccxt WebSocket | none |
| arb_execute | Coinbase | TRADE_ENABLED |

## Copy-trade tiers

| Tier | Size | Execution |
|------|------|-----------|
| AUTO | 1% of equity | Bot executes if TRADE_ENABLED=true |
| SHOW-LOW | 5% of equity | Shown in Discord, user copies manually |
| SHOW-HIGH | 25% of equity | Shown in Discord, user copies manually |

## .env keys
```
# Required
ANTHROPIC_API_KEY=
DISCORD_BOT_TOKEN=
DISCORD_ALERT_CHANNEL_ID=
COINBASE_API_KEY=
COINBASE_API_SECRET=          # PEM as single line with \n escaped
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
TV_WEBHOOK_SECRET=            # matches TradingView alert URL param
DEERTICK_MODEL=claude-sonnet-4-6

# Optional
NGROK_AUTHTOKEN=              # if unset, run: ngrok http 8080 manually
ROBINHOOD_USERNAME=
ROBINHOOD_PASSWORD=

# Trading gates
TRADE_ENABLED=false           # flip to true after 2+ profitable Alpaca cycles
ARB_MAX_TRADE_USD=50
ARB_MIN_PROFIT_PCT=0.10
ARB_AUTO_PCT=0.01
ARB_SHOW_MIN_PCT=0.05
ARB_SHOW_MAX_PCT=0.25
REBALANCE_INTERVAL_SEC=300
ARB_SCAN_INTERVAL_SEC=120
THRESHOLD_POLL_SEC=60
```

## Model switching
```bash
# In .env
DEERTICK_MODEL=claude-haiku-4-5-20251001    # cheap, ~$0.0001/call
DEERTICK_MODEL=claude-sonnet-4-6             # default, ~$0.003/call
DEERTICK_MODEL=claude-opus-4-6              # richest context, ~$0.015/call

# Or per-session override
./boot.sh --model=claude-opus-4-6
```

## Webhook (TradingView)

URL: `https://rene-aristolochiaceous-ayanna.ngrok-free.dev/webhook`
```json
{"symbol":"BTC-USD","side":"buy","price":{{close}},"secret":"deerflow-tv-secret-2026"}
```

## Quantized memory

4096-bit (512-byte) ring buffer at `~/.deer-tick-qmem.json`.
Stores last 16 market signals, 8 fields × 4 bits each.
Injected into every rebalance prompt as longitudinal context.
