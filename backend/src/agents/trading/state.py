"""TradingState — domain state for the tick-switch multi-agent trading DAG."""
from __future__ import annotations

import time
from typing import Annotated, Any, NotRequired, TypedDict

from src.agents.thread_state import ThreadState


class AnalystReport(TypedDict):
    analyst: str       # "market" | "fundamentals" | "news" | "social"
    content: str
    confidence: str    # "HIGH" | "MEDIUM" | "LOW"
    timestamp: float


class ExecutionPlan(TypedDict):
    venue: str              # "coinbase" | "alpaca" | "both"
    symbol: str
    side: str               # "BUY" | "SELL"
    notional_usd: float
    order_type: str         # "market" | "limit"
    reason: str
    copy_trade_auto: float
    copy_trade_show_low: float
    copy_trade_show_high: float


def _merge_reports(existing: list | None, new: list | None) -> list:
    if existing is None:
        return new or []
    if new is None:
        return existing
    merged = {r["analyst"]: r for r in existing}
    for r in (new or []):
        merged[r["analyst"]] = r
    return list(merged.values())


class TradingState(ThreadState):
    """ThreadState extended with the full trading DAG context."""

    # injected by the scheduler before graph invocation
    trigger: NotRequired[dict[str, Any] | None]
    market_data: NotRequired[dict[str, Any] | None]
    # {"type": "rebalance" | "arb" | "threshold", "capital_usd": float, ...}

    # phase tracking
    phase: NotRequired[str | None]
    # "analysis"|"debate"|"planning"|"awaiting_approval"|"execution"|"done"

    # analyst fan-out layer
    analyst_reports: Annotated[list[AnalystReport], _merge_reports]

    # debate / synthesis layer
    debate_thesis: NotRequired[str | None]

    # planner / risk layer
    verdict: NotRequired[str | None]        # "APPROVE" | "VETO" | "HOLD"
    confidence: NotRequired[str | None]     # "HIGH" | "MEDIUM" | "LOW"
    execution_plan: NotRequired[ExecutionPlan | None]
    risk_notes: NotRequired[str | None]
    formatted_recommendation: NotRequired[str | None]

    # HITL gate
    approval_id: NotRequired[str | None]    # short UUID set when verdict=APPROVE
    approval_granted: NotRequired[bool | None]

    # execution result
    execution_result: NotRequired[str | None]
    execution_error: NotRequired[str | None]
