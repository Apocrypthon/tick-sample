"""analysts.py — parallel fan-out: market / fundamentals / news / social."""
from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from src.models import create_chat_model
from src.agents.trading.prompts import (
    FUNDAMENTALS_ANALYST_PROMPT,
    MARKET_ANALYST_PROMPT,
    NEWS_ANALYST_PROMPT,
    SOCIAL_ANALYST_PROMPT,
)
from src.agents.trading.state import AnalystReport, TradingState

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS  = 3
_TOOL_RESULT_CAP = 4000

_PROMPTS = {
    "market":       MARKET_ANALYST_PROMPT,
    "fundamentals": FUNDAMENTALS_ANALYST_PROMPT,
    "news":         NEWS_ANALYST_PROMPT,
    "social":       SOCIAL_ANALYST_PROMPT,
}


# ── quant context formatter ───────────────────────────────────────────────

def _format_quant_context(sig_ctx: dict) -> str:
    if not sig_ctx:
        return ""
    lines = ["<quant_context>"]
    if sig_ctx.get("narrative"):
        lines.append(f"Market: {sig_ctx['narrative']}")
    if sig_ctx.get("regime"):
        lines.append(f"Regime: {sig_ctx['regime']}")
    if sig_ctx.get("fear_greed") is not None:
        lines.append(f"Fear/Greed: {sig_ctx['fear_greed']}/100")
    for pid, s in sig_ctx.get("signals", {}).items():
        lines.append(
            f"{pid}: score={s['score']:+.3f} RSI={s['rsi']:.0f} "
            f"z={s['z']:+.2f} vol_regime={s['vol_regime']} "
            f"kalman_5bar={s['kalman_forecast']:.4f}"
        )
    lines.append("</quant_context>")
    return "\n".join(lines)


# ── MCP tool lifecycle ────────────────────────────────────────────────────

@asynccontextmanager
async def _mcp_tools() -> AsyncIterator[list[BaseTool]]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.debug("langchain-mcp-adapters not installed — no MCP tools")
        yield []
        return
    try:
        from src.config.extensions_config import ExtensionsConfig
        from src.mcp.client import build_servers_config
        from src.mcp.oauth import build_oauth_tool_interceptor, get_initial_oauth_headers

        ext = ExtensionsConfig.from_file()
        cfg = build_servers_config(ext)
        if not cfg:
            yield []
            return

        oauth_headers = await get_initial_oauth_headers(ext)
        for name, hdr in oauth_headers.items():
            if name in cfg and cfg[name].get("transport") in ("sse", "http"):
                headers = dict(cfg[name].get("headers", {}))
                headers["Authorization"] = hdr
                cfg[name]["headers"] = headers

        interceptors = []
        oauth_int = build_oauth_tool_interceptor(ext)
        if oauth_int is not None:
            interceptors.append(oauth_int)

        async with MultiServerMCPClient(cfg, tool_interceptors=interceptors) as client:
            tools = await client.get_tools()
            logger.info("MCP: %d tool(s) available for analysts", len(tools))
            yield tools
    except Exception as exc:
        logger.warning("MCP tools unavailable: %s", exc)
        yield []


# ── tool execution ────────────────────────────────────────────────────────

async def _exec_tool(tools_map: dict[str, BaseTool], tc: dict) -> str:
    tool = tools_map.get(tc["name"])
    if not tool:
        return f"Unknown tool: {tc['name']}"
    try:
        result = await tool.ainvoke(tc["args"])
        text = str(result)
        if len(text) > _TOOL_RESULT_CAP:
            text = text[:_TOOL_RESULT_CAP] + f"\n... [truncated, {len(str(result))} total]"
        return text
    except Exception as e:
        return f"Tool error ({tc['name']}): {e}"


# ── confidence extractor ──────────────────────────────────────────────────

def _extract_confidence(text: str) -> str:
    for line in text.splitlines():
        if "CONFIDENCE:" in line:
            val = line.split(":", 1)[1].strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                return val
    return "MEDIUM"


# ── single analyst runner ─────────────────────────────────────────────────

async def _call_analyst(
    role: str, context: str, model, tools: list[BaseTool],
) -> AnalystReport:
    try:
        tools_map = {t.name: t for t in tools}
        bound = model.bind_tools(tools) if tools else model
        msgs: list = [
            SystemMessage(content=_PROMPTS[role]),
            HumanMessage(content=context),
        ]
        resp = None
        for round_i in range(MAX_TOOL_ROUNDS):
            resp = await bound.ainvoke(msgs)
            if not getattr(resp, "tool_calls", None):
                break
            msgs.append(resp)
            names = []
            for tc in resp.tool_calls:
                result = await _exec_tool(tools_map, tc)
                msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                names.append(tc["name"])
            logger.info("Analyst[%s] tool round %d: %s", role, round_i + 1, names)
        else:
            resp = await model.ainvoke(msgs)

        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        conf = _extract_confidence(content)
        tool_count = sum(1 for m in msgs if isinstance(m, ToolMessage))
        logger.info(
            "Analyst[%s] done  confidence=%s  tools_used=%d  chars=%d",
            role, conf, tool_count, len(content),
        )
        return AnalystReport(
            analyst=role, content=content, confidence=conf, timestamp=time.time(),
        )
    except Exception as exc:
        logger.exception("Analyst[%s] failed", role)
        return AnalystReport(
            analyst=role,
            content=f"[ANALYST ERROR: {exc}]",
            confidence="LOW",
            timestamp=time.time(),
        )


# ── context builder ───────────────────────────────────────────────────────

def _format_quotes(quotes: dict) -> str:
    if not quotes:
        return "No live quotes available."
    lines = []
    for pid, q in quotes.items():
        lines.append(
            f"  {pid}: bid=${q['bid']:.2f}  ask=${q['ask']:.2f}  "
            f"mid=${q['mid']:.2f}  spread={q['spread_bps']}bps"
        )
    return "\n".join(lines)


def _format_portfolios(portfolios: dict) -> str:
    if not portfolios:
        return "No portfolio data available."
    lines = []
    for venue, summary in portfolios.items():
        lines.append(f"── {venue.upper()} ──\n{summary}")
    return "\n".join(lines)


def _build_context(
    trigger: dict | None,
    market_data: dict | None = None,
    signal_context: dict | None = None,
) -> str:
    if not trigger:
        return "Perform a general portfolio review. No specific trigger context provided."

    t = trigger.get("type", "unknown")
    lines = [f"TRIGGER TYPE: {t.upper()}"]

    if t == "rebalance":
        lines += [
            f"Capital base: ${trigger.get('capital_usd', 0):.2f}",
            f"Quantized memory context:\n{trigger.get('qmem_ctx', 'N/A')}",
            "Task: Identify the single most optimal rebalancing trade right now.",
        ]
    elif t == "arb":
        opp = trigger.get("opportunity", {})
        lines += [
            f"Triangle: {opp.get('name', 'unknown')}",
            f"Profit estimate: {opp.get('profit_pct', 0):+.4f}%",
            f"Capital base: ${trigger.get('capital_usd', 0):.2f}",
            f"Legs: {opp.get('legs', [])}",
            f"Age since detection: {trigger.get('age_seconds', 0):.1f}s",
            "Task: Assess viability of this triangular arb. Is the spread still live?",
        ]
    elif t == "threshold":
        lines += [
            f"Symbol: {trigger.get('symbol', 'unknown')}",
            f"Price move: {trigger.get('pct_move', 0):+.1f}%",
            f"Current price: ${trigger.get('current_price', 0):.4f}",
            f"Baseline: ${trigger.get('baseline_price', 0):.4f}",
            "Task: Is this price move a tradeable opportunity or a risk to existing positions?",
        ]
    else:
        lines.append(f"Raw trigger: {trigger}")

    if market_data:
        lines.append("\n━━━ LIVE MARKET DATA (real-time from hydrate node) ━━━")
        lines.append(_format_quotes(market_data.get("quotes", {})))
        lines.append("\n━━━ PORTFOLIO SNAPSHOTS ━━━")
        lines.append(_format_portfolios(market_data.get("portfolios", {})))
        lines.append("━━━ END LIVE DATA ━━━")

    if signal_context:
        quant = _format_quant_context(signal_context)
        if quant:
            lines.append(f"\n{quant}")

    return "\n".join(lines)


# ── node entry point ──────────────────────────────────────────────────────

async def run_analysts(state: TradingState, config: RunnableConfig) -> dict:
    """Parallel fan-out to 4 specialist analysts with hydrated context + MCP tools."""
    trigger        = state.get("trigger")
    market_data    = state.get("market_data")
    signal_context = state.get("signal_context") or {}
    context        = _build_context(trigger, market_data, signal_context)
    t_type         = trigger.get("type", "none") if trigger else "none"
    model          = create_chat_model(thinking_enabled=False)

    async with _mcp_tools() as tools:
        logger.info(
            "Trading DAG: launching 4 analysts  trigger=%s  hydrated=%s  mcp_tools=%d",
            t_type, bool(market_data), len(tools),
        )
        reports = await asyncio.gather(
            _call_analyst("market",       context, model, tools),
            _call_analyst("fundamentals", context, model, tools),
            _call_analyst("news",         context, model, tools),
            _call_analyst("social",       context, model, tools),
        )

    return {"analyst_reports": list(reports), "phase": "debate"}
