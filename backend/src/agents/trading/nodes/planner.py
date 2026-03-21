"""planner.py — risk-gated trade planner: thesis → verdict + execution plan."""
from __future__ import annotations

import logging
import uuid
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.models import create_chat_model
from src.agents.trading.prompts import PLANNER_PROMPT
from src.agents.trading.state import ExecutionPlan, TradingState

logger = logging.getLogger(__name__)


# ── Structured output schema (Pydantic, internal to planner) ────────────────

class PlannerDecision(BaseModel):
    """Risk-validated trade decision from the planner agent."""

    verdict: Literal["APPROVE", "VETO", "HOLD"] = Field(
        description="Final decision: APPROVE to trade, VETO to block, HOLD to wait.",
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        description="Confidence level in the verdict.",
    )
    symbol: str = Field(
        default="N/A",
        description="Trading symbol, e.g. BTC-USD, AAPL. N/A if no trade.",
    )
    venue: Literal["coinbase", "alpaca", "both", "N/A"] = Field(
        default="N/A",
        description="Execution venue.",
    )
    side: Literal["BUY", "SELL", "N/A"] = Field(
        default="N/A",
        description="Trade direction.",
    )
    notional_usd: float = Field(
        default=0.0,
        description="Full recommended notional size in USD.",
    )
    reason: str = Field(
        default="",
        description="One-sentence rationale for the verdict.",
    )
    risk_notes: str = Field(
        default="OK",
        description="Risk concerns, veto/hold rationale, or OK if clean.",
    )
    copy_auto_usd: float = Field(
        default=0.0,
        description="Auto-execute tier size in USD (1% of capital).",
    )
    copy_show_low_usd: float = Field(
        default=0.0,
        description="Low copy-trade suggestion tier in USD (5% of capital).",
    )
    copy_show_high_usd: float = Field(
        default=0.0,
        description="High copy-trade suggestion tier in USD (25% of capital).",
    )


# ── Hard veto overrides (post-LLM validation) ───────────────────────────────

def _apply_hard_vetoes(
    decision: PlannerDecision,
    trigger: dict,
    capital: float,
) -> PlannerDecision:
    """Override LLM verdict when hard risk rules are violated."""
    vix = float(trigger.get("vix", 0.0))
    ttype = trigger.get("type", "")

    overrides: dict = {}

    if decision.verdict == "APPROVE" and vix >= 35:
        overrides = dict(
            verdict="VETO",
            risk_notes=f"Hard veto: VIX proxy {vix:.1f} >= 35 threshold.",
        )
    elif decision.verdict == "APPROVE" and decision.confidence == "LOW":
        overrides = dict(
            verdict="HOLD",
            risk_notes="Hard rule: LOW confidence downgrades APPROVE to HOLD.",
        )
    elif (
        decision.verdict == "APPROVE"
        and ttype == "arb"
        and float(trigger.get("age_seconds", 0)) > 30
    ):
        overrides = dict(
            verdict="VETO",
            risk_notes=f"Hard veto: arb age {trigger.get('age_seconds')}s > 30s limit.",
        )
    elif (
        decision.verdict == "APPROVE"
        and capital > 0
        and decision.notional_usd > capital * 0.25
    ):
        overrides = dict(
            verdict="VETO",
            risk_notes=(
                f"Hard veto: notional ${decision.notional_usd:.2f} exceeds "
                f"25% of ${capital:.2f} capital."
            ),
        )

    if overrides:
        logger.warning("Hard veto override: %s", overrides["risk_notes"])
        return decision.model_copy(update=overrides)
    return decision


# ── Discord-ready recommendation formatter ──────────────────────────────────

def _format_recommendation(
    decision: PlannerDecision,
    trigger: dict,
    approval_id: str | None,
) -> str:
    emoji = {"APPROVE": "✅", "VETO": "🚫", "HOLD": "⏸️"}.get(decision.verdict, "❓")
    t = trigger.get("type", "unknown").upper()
    lines = [
        f"## {emoji} {t} ANALYSIS — {decision.verdict} ({decision.confidence} confidence)",
        "",
        f"**Symbol:** {decision.symbol}",
        f"**Side:** {decision.side}",
        f"**Reason:** {decision.reason}",
        f"**Risk Notes:** {decision.risk_notes}",
        "",
    ]
    if decision.verdict == "APPROVE" and approval_id:
        lines += [
            "**Copy-Trade Tiers:**",
            f"  🤖 AUTO  (1%):    {decision.side} {decision.symbol} **${decision.copy_auto_usd:.2f}**",
            f"  📋 SHOW  (5%):   {decision.side} {decision.symbol} ${decision.copy_show_low_usd:.2f}",
            f"  📋 SHOW (25%): {decision.side} {decision.symbol} ${decision.copy_show_high_usd:.2f}",
            "",
            f"⚡ React ✅ to execute AUTO tier  |  ID: `{approval_id}`",
            "❌ React ❌ to reject",
        ]
    elif decision.verdict == "VETO":
        lines.append(f"**Veto reason:** {decision.risk_notes}")
    else:
        lines.append("**No trade at this time.** Monitoring continues.")
    return "\n".join(lines)


# ── Node entry point ────────────────────────────────────────────────────────

_STRUCTURED_PROMPT = PLANNER_PROMPT.split("Sizing tiers")[0].rstrip()
# Strip the trailing block-format instructions — structured output replaces them.

async def run_planner(state: TradingState, config: RunnableConfig) -> dict:
    """Produce a risk-validated verdict and execution plan."""
    thesis = state.get("debate_thesis", "")
    trigger = state.get("trigger") or {}
    capital = float(trigger.get("capital_usd") or 1000.0)
    vix = float(trigger.get("vix", 0.0))

    human = (
        f"Capital base: ${capital:.2f}\n"
        f"VIX proxy (VIXY): {vix:.2f}\n"
        f"Trigger type: {trigger.get('type', 'unknown')}\n\n"
        f"Debate Thesis:\n{thesis}\n\n"
        "Produce the risk-validated execution plan."
    )

    model = create_chat_model(thinking_enabled=False)
    structured = model.with_structured_output(PlannerDecision)

    try:
        decision: PlannerDecision = await structured.ainvoke(
            [SystemMessage(content=_STRUCTURED_PROMPT), HumanMessage(content=human)]
        )
    except Exception as exc:
        logger.exception("Planner structured call failed")
        decision = PlannerDecision(
            verdict="HOLD",
            confidence="LOW",
            risk_notes=f"Planner error: {exc}",
        )

    decision = _apply_hard_vetoes(decision, trigger, capital)

    plan: ExecutionPlan | None = None
    approval_id: str | None = None

    if decision.verdict == "APPROVE":
        approval_id = str(uuid.uuid4())[:8]
        plan = ExecutionPlan(
            venue=decision.venue,
            symbol=decision.symbol,
            side=decision.side,
            notional_usd=decision.notional_usd,
            order_type="market",
            reason=decision.reason,
            copy_trade_auto=decision.copy_auto_usd,
            copy_trade_show_low=decision.copy_show_low_usd,
            copy_trade_show_high=decision.copy_show_high_usd,
        )

    recommendation = _format_recommendation(decision, trigger, approval_id)
    logger.info(
        "Planner: verdict=%s confidence=%s approval_id=%s",
        decision.verdict, decision.confidence, approval_id,
    )

    return {
        "verdict": decision.verdict,
        "confidence": decision.confidence,
        "execution_plan": plan,
        "risk_notes": decision.risk_notes,
        "formatted_recommendation": recommendation,
        "approval_id": approval_id,
        "phase": "awaiting_approval" if decision.verdict == "APPROVE" else "done",
    }
