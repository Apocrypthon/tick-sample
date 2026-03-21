# tick-switch

LangGraph-based multi-agent trading system with adversarial debate, HITL execution gating, and multi-venue broker support.

## Architecture
┌─────────────────────────────────────────────────────────────┐
│                    TRADING DAG (LangGraph)                   │
│                                                             │
│  ┌──────────┐   ┌────────────────────────┐   ┌───────────┐ │
│  │ HYDRATE  │──▶│     ANALYSTS (4x ∥)    │──▶│  DEBATE   │ │
│  │ live     │   │ market | fund | news | │   │ bull ∥    │ │
│  │ quotes + │   │ social — all receive   │   │ bear →    │ │
│  │ portfolio│   │ real market data        │   │ synthesis │ │
│  └──────────┘   └────────────────────────┘   └─────┬─────┘ │
│                                                    │       │
│                                              ┌─────▼─────┐ │
│                                              │  PLANNER  │ │
│                                              │ risk gate │ │
│                                              │ copy-trade│ │
│                                              │ tiers     │ │
│                                              └─────┬─────┘ │
└────────────────────────────────────────────────────┼───────┘
│
┌──────────▼──────────┐
│   DISCORD HITL      │
│   ✅ approve → exec │
│   ❌ reject → skip  │
└──────────┬──────────┘
│
┌──────────▼──────────┐
│   EXECUTION BRIDGE  │
│  coinbase | alpaca  │
│  paper / live mode  │
└─────────────────────┘




## DAG Nodes

| Node | Purpose | LLM Calls |
|------|---------|-----------|
| **hydrate** | Fetches live quotes (Coinbase bid/ask/mid) + portfolio snapshots (Coinbase + Alpaca) | 0 |
| **analysts** | 4 parallel specialists: market, fundamentals, news, social — all grounded in hydrated data | 4 (parallel) |
| **debate** | Adversarial bull/bear researchers (parallel) → synthesis manager resolves conflicts | 3 (2∥ + 1) |
| **planner** | Risk-gated trade planner: hard veto rules, copy-trade tier sizing | 1 |

**Total per pipeline run:** 8 LLM calls, ~5 wall-clock serial calls (hydrate is I/O only).

## Trigger Types

The `AlertScheduler` fires three concurrent loops that feed triggers into the DAG:

| Loop | Interval | Trigger |
|------|----------|---------|
| **rebalance** | 300s | Portfolio rebalance scan with quantized memory context |
| **threshold** | 60s | Price move > ±3% from baseline (doubles threshold routes through DAG) |
| **arb** | 60s | Triangular arbitrage opportunity > 0.10% profit |

## Risk Engine (Deterministic, Pre-Trade)

Hard veto rules in the planner — LLM cannot bypass:
- VIXY ≥ 35 → veto all buys
- Thesis confidence LOW → HOLD, never APPROVE
- Arb age > 30s → VETO (spread likely closed)
- Notional > 25% of portfolio → reduce or VETO

## Copy-Trade Tiers

| Tier | Size | Behavior |
|------|------|----------|
| AUTO (1%) | `capital × 0.01` | Micro-execute on ✅ reaction |
| SHOW LOW (5%) | `capital × 0.05` | Displayed as suggestion |
| SHOW HIGH (25%) | `capital × 0.25` | Displayed as suggestion |

## Execution Safety

- `TRADE_ENABLED=false` (default): all executions are dry runs
- Alpaca is always paper mode (`ALPACA_PAPER=true`)
- Coinbase is live when `TRADE_ENABLED=true` — the only path to real money
- Discord ✅ reaction required for any execution (HITL gate)

## Broker Adapters

| Venue | Library | Asset Class |
|-------|---------|-------------|
| Coinbase | `coinbase-advanced-py` | Crypto (BTC, ETH, SOL, etc.) |
| Alpaca | `alpaca-py` | Equities (paper) |
| Robinhood | `robin-stocks` | Equities + crypto (read-only for portfolio/quotes) |

## Setup

```bash
cd backend
cp .env.template .env
# Fill in API keys in .env
uv sync
uv run python discord\_bridge.py
Project Structure


backend/
├── discord\_bridge.py          # Entry point — boots all systems
├── config.yaml                # Model, tools, sandbox, memory config
├── src/
│   ├── agents/trading/
│   │   ├── graph.py           # DAG wiring: hydrate → analysts → debate → planner
│   │   ├── state.py           # TradingState with Annotated reducers
│   │   ├── prompts.py         # All system prompts (analysts, bull/bear, planner)
│   │   └── nodes/
│   │       ├── hydrate.py     # Live data fetcher (quotes + portfolios)
│   │       ├── analysts.py    # 4x parallel analyst fan-out
│   │       ├── debate.py      # Adversarial bull/bear → synthesis
│   │       ├── execution.py   # Venue-routed order execution
│   │       └── planner.py     # Risk-gated trade planner
│   ├── arb/                   # Triangular arbitrage scanner
│   ├── channels/              # Discord channel + message bus
│   ├── scheduler/             # Alert loops (rebalance, threshold, arb)
│   ├── tools/                 # Broker tool wrappers
│   └── mcp/                   # MCP client for external data servers
License
See LICENSE.
