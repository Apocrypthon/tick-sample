
> Live demo: `deer-tick#7045` on Discord, tunnel at `rene-aristolochiaceous-ayanna.ngrok-free.dev`

---

## Table of Contents

1. [What It Is](#1-what-it-is)
2. [How It Works](#2-how-it-works)
3. [Architecture](#3-architecture)
4. [Signal & Risk Stack](#4-signal--risk-stack)
5. [Installation — aarch64 / ChromeOS Penguin](#5-installation--aarch64--chromeos-penguin)
6. [Installation — x86\_64 Linux (Ubuntu/Debian)](#6-installation--x8664-linux-ubuntudebian)
7. [Installation — RHEL / CentOS / Fedora](#7-installation--rhel--centos--fedora)
8. [Configuration](#8-configuration)
9. [Running the Service](#9-running-the-service)
10. [TradingView Webhook Integration](#10-tradingview-webhook-integration)
11. [Discord Control Interface](#11-discord-control-interface)
12. [Testing](#12-testing)
13. [Deployment Architecture](#13-deployment-architecture)
14. [References](#14-references)

---

## 1. What It Is

tick-sample is a personal autonomous trading agent that connects **TradingView alerts** to **live broker execution** via a multi-agent reasoning pipeline and a **human-in-the-loop Discord approval gate**. It is a fork of [ByteDance DeerFlow](https://github.com/bytedance/deer-flow) [^1] with the web frontend replaced entirely by a Discord bot interface and the agent reasoning core replaced with Anthropic Claude [^2].

The system is designed around three responsibilities held by distinct layers:

| Layer | Responsibility | Key modules |
|-------|---------------|-------------|
| **Signal** | Generate quantitative trade signals from live market data | `signal_engine.py`, `kalman.py`, `hmm.py`, `alpha_factors.py` |
| **Reasoning** | Multi-agent adversarial debate over signals → risk-gated verdict | `analysts.py`, `debate.py`, `planner_v2.py` |
| **Execution** | Route approved trades to brokers; HITL gate via Discord reactions | `discord_channel.py`, `hitl/gate.py`, `execution.py` |

No trade touches a broker without a human ✅ reaction in Discord. `TRADE_ENABLED=false` by default — all execution is dry-run until explicitly enabled.

---

## 2. How It Works

### 2.1 Trigger sources

Three concurrent loops generate trading triggers:

```
REBALANCE  every 300s  → portfolio scan + QuantMemory context
THRESHOLD  every  60s  → price move > ±3% from baseline (Robinhood watchlist)
ARB        every  60s  → triangular arbitrage scan (Coinbase order books)
TradingView alerts     → POST /webhook (ngrok static domain → FastAPI)
```

### 2.2 The trading DAG

Each trigger enters a five-node LangGraph DAG [^3]:

```
START
  │
  ▼
HYDRATE ──── fetches live Coinbase quotes, Alpaca + Coinbase portfolios
             runs SignalEngine + MarketIntelligence
             writes signal_context into state
  │
  ▼
ANALYSTS ─── 4× parallel LLM calls (market · fundamentals · news · social)
             each receives <quant_context> XML block from signal layer
  │
  ▼
DEBATE ───── adversarial bull/bear researchers argue in parallel
             synthesis manager resolves conflicts by weighted rules [^4]
  │
  ▼
PLANNER ──── Kelly-criterion sizing + hard veto rules (VIX ≥ 30, LOW confidence,
             arb age > 30s, notional > 25% portfolio)
             produces copy-trade tiers: AUTO 1% · SHOW-LOW 5% · SHOW-HIGH 25%
  │
  ▼
END ──────── formatted recommendation → Discord
             if APPROVE: ✅/❌ reactions added; awaiting HITL gate
```

### 2.3 Human-in-the-loop gate

When `planner_v2` returns `APPROVE`, the Discord channel:

1. Posts the formatted recommendation with ✅/❌ reactions
2. Registers `{discord_message_id: approval_id, execution_plan, chat_id}` in `_pending_approvals`
3. On ✅ reaction from an `allowed_users` ID: publishes `__EXECUTE__ <approval_id> <plan_json>` back into the message bus
4. `execution.py` routes to Coinbase (live if `TRADE_ENABLED=true`) or Alpaca (always paper)

---

## 3. Architecture

```
TradingView alert
  → ngrok static domain (pyngrok managed)
  → FastAPI webhook_server.py :8080
  → MessageBus (asyncio pub/sub)
  → DirectDispatcher → DeerTickClient
  → LangGraph agent (Claude claude-haiku-4-5-20251001)

AlertScheduler (3 async loops)
  → MessageBus
  → same path as above

Discord bot (discord.py gateway)
  → MessageBus inbound (chat messages + reactions)
  → MessageBus outbound (analysis cards + dry-run orders)

Background data systems (non-blocking, warnings-only if failing):
  ArbScanner       ccxt.pro WebSocket order books (6 symbols, 2 batches ≤4)
  OHLCVAggregator  REST polling 1m bars (Coinbase fetch_ohlcv, 60s interval)
  TTLCache         thread-safe LRU: quotes 10s · ohlcv 65s · portfolio 30s
```

The message bus, dispatcher, Discord channel, webhook server, arb scanner, OHLCV aggregator, and scheduler are all started in `discord_bridge.py:main()` and shut down cleanly on `SIGINT`/`SIGTERM`.

---

## 4. Signal & Risk Stack

### 4.1 Quantitative signal modules

| Module | Algorithm | Output |
|--------|-----------|--------|
| `forecasting/kalman.py` | Adaptive Kalman filter [^5] | level, velocity, 5-bar forecast |
| `forecasting/arima_garch.py` | ARIMA(1,1,1) + EWMA-GARCH fallback [^6] | point forecast, volatility surface |
| `features/alpha_factors.py` | Pure-numpy RSI, z-score, vol-ratio, Amihud illiquidity [^7] | normalised factor vector |
| `regime/hmm.py` | Gaussian HMM 4-state Viterbi [^8] | BULL / BEAR / VOLATILE / SIDEWAYS |
| `signals/signal_engine.py` | Composite weighted score | scalar ∈ [-1, +1] |
| `intelligence/market_intelligence.py` | 2-state HMM + rolling correlation + EWMA vol | Fear/Greed index, regime label |

### 4.2 Risk engine

`risk/engine.py` implements deterministic pre-trade rules that **cannot be overridden by LLM reasoning**:

```python
VIX  ≥ 30          → VETO  (hard)
confidence == LOW  → HOLD  (hard)
arb age > 30s      → VETO  (hard)
notional > 25% NAV → VETO  (hard)
drawdown ≥ 10%     → HALT  (circuit breaker)
drawdown ≥  5%     → SCALE (position size × 0.5)
```

Kelly criterion [^9] sizes each position. CVaR at 95% confidence quantifies tail risk. Copy-trade tiers are computed as fixed fractions of capital:

| Tier | Size | Execution |
|------|------|-----------|
| AUTO | capital × 0.01 | Bot executes on ✅ if `TRADE_ENABLED=true` |
| SHOW-LOW | capital × 0.05 | Displayed in Discord; user copies manually |
| SHOW-HIGH | capital × 0.25 | Displayed in Discord; user copies manually |

### 4.3 QuantMemory v2

A 1024-bit ring buffer (`scheduler/quant_memory_v2.py`) stores the last 16 market signals, each encoded as 16 fields × 4 bits. The quantized summary is injected into every rebalance DAG run as longitudinal market context, giving the LLM a compressed view of recent market history without token-expensive raw data.

---

## 5. Installation — aarch64 / ChromeOS Penguin

Tested on: Debian 12 (bookworm) aarch64, ChromeOS Penguin container, 2.7 GB RAM, 19.5 GB disk.

```bash
# 1. System deps
sudo apt update && sudo apt install -y git curl python3.12 python3.12-venv python3.12-dev

# 2. uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. Clone
git clone git@github.com:Apocrypthon/tick-sample.git
cd tick-sample/backend

# 4. Python environment
uv python pin 3.12
uv sync

# 5. Environment
cp .env.template .env
# Edit .env — minimum required keys listed in §8

# 6. Systemd service
sudo cp /path/to/tick-sample@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tick-sample@$(whoami)

# 7. Verify
sudo journalctl -u tick-sample@$(whoami) -f
```

---

## 6. Installation — x86\_64 Linux (Ubuntu/Debian)

Tested on: Ubuntu 22.04 LTS x86\_64, Ubuntu 24.04 LTS x86\_64.

```bash
# 1. System deps
sudo apt update
sudo apt install -y git curl build-essential \
    python3.12 python3.12-venv python3.12-dev \
    libssl-dev libffi-dev

# 2. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
# verify
uv --version

# 3. Clone and install
git clone git@github.com:Apocrypthon/tick-sample.git
cd tick-sample/backend
uv python pin 3.12
uv sync

# 4. Confirm ccxt.pro is present (required for arb scanner)
uv run python3 -c "import ccxt.pro; print('ccxt.pro OK')"

# 5. Environment and service — same as §5 steps 5-7
```

**Note for Ubuntu 24.04:** Python 3.12 is the system default. No PPA required. If running Ubuntu 22.04, add the deadsnakes PPA:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

---

## 7. Installation — RHEL / CentOS / Fedora

Tested on: RHEL 9.x, CentOS Stream 9, Fedora 39+.

```bash
# 1. System deps (RHEL/CentOS 9)
sudo dnf update -y
sudo dnf install -y git curl gcc gcc-c++ make \
    openssl-devel libffi-devel bzip2-devel \
    readline-devel sqlite-devel

# 2. Python 3.12 — not in default RHEL 9 repos, build from source or use pyenv
# Option A: pyenv (recommended)
curl https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc
pyenv install 3.12.13
pyenv global 3.12.13

# Option B: Fedora 38+ — Python 3.12 in default repos
sudo dnf install -y python3.12 python3.12-devel

# 3. uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 4. SELinux consideration
# If running as a systemd service on RHEL with SELinux enforcing:
sudo setsebool -P nis_enabled 1
# Or create a custom policy for the service socket/port 8080:
sudo semanage port -a -t http_port_t -p tcp 8080

# 5. firewalld — open webhook port if receiving from external network
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload

# 6. Clone and install
git clone git@github.com:Apocrypthon/tick-sample.git
cd tick-sample/backend
uv python pin 3.12
uv sync

# 7. Systemd — RHEL uses /etc/systemd/system same as Debian
sudo cp tick-sample@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now tick-sample@$(whoami)

# 8. Verify
sudo journalctl -u tick-sample@$(whoami) -f
```

**RHEL note:** `python-devel` and `gcc` are required to compile any C extensions in the dependency chain. The `uv sync` step handles the rest without root.

---

## 8. Configuration

All configuration lives in `backend/.env`. Copy from `.env.template` and fill in:

```bash
# ── Model ─────────────────────────────────────────────────────────────────────
TICK_SAMPLE_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=sk-ant-...

# ── Discord ───────────────────────────────────────────────────────────────────
DISCORD_BOT_TOKEN=                     # Bot tab → Reset Token (NOT OAuth secret)
DISCORD_ALERT_CHANNEL_ID=              # Right-click channel → Copy ID

# ── ngrok (static domain — configure once in ngrok dashboard) ─────────────────
NGROK_AUTHTOKEN=                       # dashboard.ngrok.com/get-started/your-authtoken
NGROK_DOMAIN=your-static-domain.ngrok-free.dev

# ── TradingView webhook ───────────────────────────────────────────────────────
TV_WEBHOOK_SECRET=your-secret-here     # must match JSON payload "secret" field

# ── Coinbase Advanced Trade ───────────────────────────────────────────────────
COINBASE_API_KEY=
COINBASE_API_SECRET=                   # PEM key — escape newlines as \n

# ── Alpaca (paper by default) ─────────────────────────────────────────────────
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_PAPER=true

# ── Robinhood (optional — used for VIX/price baseline; gracefully skipped if absent)
ROBINHOOD_USERNAME=
ROBINHOOD_PASSWORD=
ROBINHOOD_MFA_CODE=

# ── Execution gate ────────────────────────────────────────────────────────────
TRADE_ENABLED=false                    # flip to true only after Alpaca paper cycles confirm
```

`backend/config.yaml` controls model parameters, tool groups, sandbox, memory, and checkpointer. The active model is resolved from `$TICK_SAMPLE_MODEL`.

---

## 9. Running the Service

```bash
# Start
sudo systemctl enable --now tick-sample@$(whoami)

# Status
sudo systemctl status tick-sample@$(whoami)

# Live logs
sudo journalctl -u tick-sample@$(whoami) -f

# Restart after .env change
sudo systemctl restart tick-sample@$(whoami)

# Stop
sudo systemctl stop tick-sample@$(whoami)
```

**Expected healthy startup sequence:**

```
INFO  ArbScanner started — 6 symbols in 2 batches of ≤4
INFO  OHLCVAggregator started — 3 symbols @ 1m
INFO  HITLGate initialized (timeout=600s)
INFO  Webhook server started on :8080
INFO  ngrok tunnel: https://<your-domain>.ngrok-free.dev
INFO  TradingView webhook URL: https://<your-domain>.ngrok-free.dev/webhook
INFO  DirectDispatcher started (max_concurrency=3)
INFO  Discord channel started
INFO  AlertScheduler started — rebalance=300s threshold=60s arb=60s
INFO  DeerTick agent running -- Ctrl+C to stop
INFO  discord.gateway -- connected to Gateway
INFO  Discord bot: deer-tick#XXXX
```

---

## 10. TradingView Webhook Integration

The static ngrok domain means the webhook URL **never changes** — configure TradingView once.

### Alert setup

1. Open any chart → **Alerts** (clock icon) → **Create Alert**
2. Set your condition (any indicator, or Pine Script `alertcondition(true)` for testing)
3. **Notifications tab** → enable **Webhook URL** → paste:
   ```
   https://<your-domain>.ngrok-free.dev/webhook
   ```
4. **Message** field — paste exactly:
   ```json
   {"symbol":"{{ticker}}","side":"BUY","price":"{{close}}","secret":"YOUR_TV_WEBHOOK_SECRET","indicator":"{{plot_0}}","timeframe":"{{interval}}","message":"{{ticker}} {{close}} @ {{time}}"}
   ```
   Replace `YOUR_TV_WEBHOOK_SECRET` with the value from `.env`.

5. The `secret` field is a fixed string — not a TradingView placeholder. This is by design: TradingView has no API for programmatic webhook registration; the static domain + shared secret is the correct pattern [^10].

### Verify a hit

```bash
curl -s https://<your-domain>.ngrok-free.dev/health
# → {"status":"ok","bus_ready":true}

# Watch live
sudo journalctl -u tick-sample@$(whoami) -f | grep -E "TradingView|routed|inbound"
```

---

## 11. Discord Control Interface

The bot listens in any channel it has access to. Send messages directly to the bot or in the alert channel:

| Interaction | Effect |
|-------------|--------|
| Any message | Routed to Claude for analysis |
| ✅ on an APPROVE card | Executes AUTO-tier trade (if `TRADE_ENABLED=true`) |
| ❌ on an APPROVE card | Rejects the trade; logs to journal |
| `/new` | Start a fresh conversation thread |
| `/status` | Print active thread ID |
| `/help` | Print command list |

The bot posts rebalance analysis cards every 300s and threshold alerts whenever a watched price moves ≥ 3% from its last recorded baseline.

---

## 12. Testing

```bash
cd ~/tick-sample/backend

# Full unit test suite (29 tests)
uv run pytest tests/test_core_modules.py -v

# Smoke test — all signal modules
uv run python3 - << 'EOF'
import sys; sys.path.insert(0, "src")
from forecasting.kalman import AdaptiveKalmanTracker
from signals.signal_engine import SignalEngine
from risk.engine import RiskEngine
import numpy as np

kf = AdaptiveKalmanTracker()
for p in np.linspace(60000, 65000, 30): s = kf.update(float(p))
print(f"Kalman  level={s.level:.0f} vel={s.velocity:+.2f} f5={s.forecast_5:.0f}")

se = SignalEngine()
prices = np.random.randn(200).cumsum() + 1000
score = se.compute(prices, volume=np.abs(np.random.randn(200)) * 1e6)
print(f"Signal  score={score.composite:+.4f} regime={score.vol_regime}")

print("ALL OK")
EOF

# Walk-forward backtest
uv run python3 - << 'EOF'
import sys, numpy as np; sys.path.insert(0, "src")
from backtest.walk_forward import WalkForwardBacktest
wf = WalkForwardBacktest(train_bars=100, test_bars=50, step_bars=50)
r  = wf.run(60000 + np.cumsum(np.random.randn(500)*100),
            lambda h: float(np.sign(h[-1]-h[-20])) if len(h)>=20 else 0.0)
print(r.summary())
EOF
```

---

## 13. Deployment Architecture

```
ChromeOS Penguin (aarch64, 2.7G RAM)
├── systemd tick-sample@_.service
│   └── discord_bridge.py (main event loop)
│       ├── ArbScanner          (ccxt.pro WebSocket, 2 async tasks)
│       ├── OHLCVAggregator     (REST polling, 3 async tasks)
│       ├── pyngrok tunnel      (static domain, auto-reconnect)
│       ├── uvicorn :8080       (FastAPI webhook receiver, daemon thread)
│       ├── DirectDispatcher    (asyncio, max_concurrency=3)
│       ├── DiscordChannel      (discord.py gateway, separate thread)
│       └── AlertScheduler      (3 async loops: rebalance/threshold/arb)
│
└── SQLite checkpoints.db       (LangGraph conversation state)
```

Memory footprint at steady state: ~241 MB RSS. No swap required.
Disk usage: ~2.9 GB (includes Python env and all deps).

---

## 14. References

[^1]: ByteDance DeerFlow (2024). *Open-source multi-agent research assistant*. https://github.com/bytedance/deer-flow. License: MIT.

[^2]: Anthropic (2025). *Claude claude-haiku-4-5-20251001 model card*. https://www.anthropic.com/claude. Accessed March 2026.

[^3]: LangChain (2024). *LangGraph: Build stateful, multi-actor applications with LLMs*. https://langchain-ai.github.io/langgraph/. License: MIT.

[^4]: Xiao, Y., Sun, E., Luo, D., & Wang, W. (2025). *TradingAgents: Multi-Agents LLM Financial Trading Framework*. arXiv:2412.20138. Used as reference architecture for adversarial debate pattern.

[^5]: Kalman, R.E. (1960). *A New Approach to Linear Filtering and Prediction Problems*. Journal of Basic Engineering, 82(1), 35–45. Adaptive variant follows Akhlaghi et al. (2017) noise covariance estimation.

[^6]: Box, G.E.P., Jenkins, G.M., & Reinsel, G.C. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley. Fixed (1,1,1) order selected per Hyndman & Khandakar (2008) recommendation for short-horizon forecasting.

[^7]: Amihud, Y. (2002). *Illiquidity and stock returns: cross-section and time-series effects*. Journal of Financial Markets, 5(1), 31–56. Alpha158 factor set inspired by Microsoft Qlib (Yang et al., 2020, arXiv:2009.11189).

[^8]: Rabiner, L.R. (1989). *A tutorial on hidden Markov models and selected applications in speech recognition*. Proceedings of the IEEE, 77(2), 257–286. 4-state HMM with Viterbi decode via hmmlearn 0.3.x.

[^9]: Kelly, J.L. (1956). *A New Interpretation of Information Rate*. Bell System Technical Journal, 35(4), 917–926. CVaR implementation follows Rockafellar & Uryasev (2000).

[^10]: ngrok Inc. (2024). *Static domains — reserve a permanent address for your tunnel*. https://ngrok.com/docs/network-edge/domains/. Accessed March 2026.

---

## License

MIT — see [LICENSE](LICENSE).

Original DeerFlow code © 2025 ByteDance Ltd. and/or its affiliates.
Modifications © 2025-2026 Apocrypthon.
