"""analysts.py — parallel fan-out: market / fundamentals / news / social.

Receives hydrated market_data from the hydrate node. Each analyst can
optionally invoke MCP tools (web search, OpenBB, etc.) for active research
via a bounded tool-calling loop.
"""
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

MAX_TOOL_ROUNDS = 3
_TOOL_RESULT_CAP = 4000  # chars per tool result to prevent context blowup

_PROMPTS = {
    "market": MARKET_ANALYST_PROMPT,
    "fundamentals": FUNDAMENTALS_ANALYST_PROMPT,
    "news": NEWS_ANALYST_PROMPT,
    "social": SOCIAL_ANALYST_PROMPT,
}


# ── MCP tool lifecycle ────────────────────────────────────────────────────


@asynccontextmanager
async def _mcp_tools() -> AsyncIterator[list[BaseTool]]:
    """Yield MCP tools with proper client lifecycle.

    MultiServerMCPClient is a context manager — tool connections only stay
    alive while the client is open.  Yields [] on any failure so callers
    never need to handle errors.
    """
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

        # Inject OAuth headers (mirrors existing get_mcp_tools logic)
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
    """Execute a single tool call, returning truncated string result."""
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


# ── analyst with tool loop ────────────────────────────────────────────────


def _extract_confidence(text: str) -> str:
    for line in text.splitlines():
        if "CONFIDENCE:" in line:
            val = line.split(":", 1)[1].strip().upper()
            if val in ("HIGH", "MEDIUM", "LOW"):
                return val
    return "MEDIUM"


async def _call_analyst(
    role: str, context: str, model, tools: list[BaseTool],
) -> AnalystReport:
    """Run one analyst with an optional bounded tool-calling loop."""
    try:
        tools_map = {t.name: t for t in tools}
        bound = model.bind_tools(tools) if tools else model
        msgs: list = [SystemMessage(content=_PROMPTS[role]), HumanMessage(content=context)]

        resp = None
        for round_i in range(MAX_TOOL_ROUNDS):
            resp = await bound.ainvoke(msgs)
            if not getattr(resp, "tool_calls", None):
                break
            # Process tool calls
            msgs.append(resp)
            names = []
            for tc in resp.tool_calls:
                result = await _exec_tool(tools_map, tc)
                msgs.append(ToolMessage(content=result, tool_call_id=tc["id"]))
                names.append(tc["name"])
            logger.info("Analyst[%s] tool round %d: %s", role, round_i + 1, names)
        else:
            # Exhausted tool rounds — force plain text response
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


# ── context building (unchanged) ─────────────────────────────────────────


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


def _build_context(trigger: dict | None, market_data: dict | None = None) -> str:
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

    return "\n".join(lines)


# ── node entry point ──────────────────────────────────────────────────────


async def run_analysts(state: TradingState, config: RunnableConfig) -> dict:
    """Parallel fan-out to 4 specialist analysts with hydrated context + MCP tools."""
    trigger = state.get("trigger")
    market_data = state.get("market_data")
    context = _build_context(trigger, market_data)
    t_type = trigger.get("type", "none") if trigger else "none"
    model = create_chat_model(thinking_enabled=False)

    async with _mcp_tools() as tools:
        logger.info(
            "Trading DAG: launching 4 analysts  trigger=%s  hydrated=%s  mcp_tools=%d",
            t_type, bool(market_data), len(tools),
        )
        reports = await asyncio.gather(
            _call_analyst("market", context, model, tools),
            _call_analyst("fundamentals", context, model, tools),
            _call_analyst("news", context, model, tools),
            _call_analyst("social", context, model, tools),
        )

    return {"analyst_reports": list(reports), "phase": "debate"}
