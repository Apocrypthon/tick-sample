"""planner_v2.py — Kelly-optimal planner with signal_context integration."""
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


class PlannerDecision(BaseModel):
    verdict:            Literal["APPROVE", "VETO", "HOLD"]
    confidence:         Literal["HIGH", "MEDIUM", "LOW"]
    symbol:             str   = "N/A"
    venue:              Literal["coinbase", "alpaca", "both", "N/A"] = "N/A"
    side:               Literal["BUY", "SELL", "N/A"] = "N/A"
    notional_usd:       float = 0.0
    reason:             str   = ""
    risk_notes:         str   = "OK"
    copy_auto_usd:      float = 0.0
    copy_show_low_usd:  float = 0.0
    copy_show_high_usd: float = 0.0


def _apply_hard_vetoes(d: PlannerDecision, trigger: dict, capital: float) -> PlannerDecision:
    vix   = float(trigger.get("vix", 0.0))
    ttype = trigger.get("type", "")
    over: dict = {}

    if d.verdict == "APPROVE" and vix >= 30.0:
        over = dict(verdict="VETO", risk_notes=f"Hard veto: VIX {vix:.1f} ≥ 30.")
    elif d.verdict == "APPROVE" and d.confidence == "LOW":
        over = dict(verdict="HOLD", risk_notes="LOW confidence → HOLD.")
    elif (d.verdict == "APPROVE" and ttype == "arb"
          and float(trigger.get("age_seconds", 0)) > 30):
        over = dict(verdict="VETO", risk_notes=f"Arb age {trigger.get('age_seconds')}s > 30s.")
    elif d.verdict == "APPROVE" and capital > 0 and d.notional_usd > capital * 0.25:
        over = dict(verdict="VETO", risk_notes=f"Notional > 25% of capital.")

    if over:
        logger.warning("Hard veto: %s", over["risk_notes"])
        return d.model_copy(update=over)
    return d


def _kelly_tiers(signal_context: dict, capital: float) -> dict[str, float]:
    sigs  = signal_context.get("signals", {})
    score = (sum(s.get("score", 0.0) for s in sigs.values()) / len(sigs)) if sigs else 0.0
    try:
        from src.risk.engine import RiskEngine
        dec = RiskEngine(capital).evaluate(
            notional=capital * 0.10,
            signal_score=score,
            realized_vol=0.80,
            vix=signal_context.get("vix", 20.0),
        )
        if dec.copy_tiers:
            return {
                "auto":      dec.copy_tiers.auto,
                "show_low":  dec.copy_tiers.show_low,
                "show_high": dec.copy_tiers.show_high,
                "kelly_opt": dec.kelly_optimal,
                "var_1d":    dec.var_1d,
                "score":     score,
            }
    except Exception as exc:
        logger.warning("Kelly tiers failed: %s", exc)
    return {
        "auto": capital*0.01, "show_low": capital*0.05,
        "show_high": capital*0.25, "kelly_opt": 0.0, "var_1d": 0.0, "score": score,
    }


def _format_rec(d: PlannerDecision, trigger: dict,
                aid: str | None, tiers: dict) -> str:
    emoji = {"APPROVE": "✅", "VETO": "🚫", "HOLD": "⏸️"}.get(d.verdict, "❓")
    t = trigger.get("type", "unknown").upper()
    lines = [
        f"## {emoji} {t} ANALYSIS — {d.verdict} ({d.confidence} confidence)", "",
        f"**Symbol:** {d.symbol}",
        f"**Side:** {d.side}",
        f"**Reason:** {d.reason}",
        f"**Risk Notes:** {d.risk_notes}",
        f"**Signal score:** {tiers['score']:+.3f}  "
        f"**Kelly optimal:** ${tiers['kelly_opt']:.2f}  "
        f"**VaR 1d:** ${tiers['var_1d']:.2f}", "",
    ]
    if d.verdict == "APPROVE" and aid:
        lines += [
            "**Copy-Trade Tiers (Kelly-adjusted):**",
            f"  🤖 AUTO  (1%):    {d.side} {d.symbol} **${tiers['auto']:.2f}**",
            f"  📋 SHOW  (5%):   {d.side} {d.symbol} ${tiers['show_low']:.2f}",
            f"  📋 SHOW (25%): {d.side} {d.symbol} ${tiers['show_high']:.2f}", "",
            f"⚡ React ✅ to execute AUTO tier  |  ID: `{aid}`",
            "❌ React ❌ to reject",
        ]
    elif d.verdict == "VETO":
        lines.append(f"**Veto reason:** {d.risk_notes}")
    else:
        lines.append("**No trade at this time.** Monitoring continues.")
    return "\n".join(lines)


_STRUCTURED_PROMPT = PLANNER_PROMPT.split("Sizing tiers")[0].rstrip()


async def run_planner(state: TradingState, config: RunnableConfig) -> dict:
    thesis         = state.get("debate_thesis", "")
    trigger        = state.get("trigger") or {}
    sig_ctx        = state.get("signal_context") or {}
    capital        = float(trigger.get("capital_usd") or 1000.0)
    vix            = float(trigger.get("vix", 20.0))

    # Build quant summary for LLM context
    sig_lines = []
    for pid, s in sig_ctx.get("signals", {}).items():
        sig_lines.append(
            f"  {pid}: score={s['score']:+.3f} RSI={s['rsi']:.0f} "
            f"z={s['z']:+.2f} vol={s['vol_regime']} kF5={s['kalman_forecast']:.4f}"
        )
    quant_block = ""
    if sig_lines:
        quant_block = "\nQuantitative Signals:\n" + "\n".join(sig_lines)
    if sig_ctx.get("narrative"):
        quant_block += f"\nMarket: {sig_ctx['narrative']}"

    human = (
        f"Capital base: ${capital:.2f}\n"
        f"VIX proxy (VIXY): {vix:.2f}\n"
        f"Trigger type: {trigger.get('type', 'unknown')}\n"
        f"{quant_block}\n\n"
        f"Debate Thesis:\n{thesis}\n\n"
        "Produce the risk-validated execution plan."
    )

    model      = create_chat_model(thinking_enabled=False)
    structured = model.with_structured_output(PlannerDecision)

    try:
        decision: PlannerDecision = await structured.ainvoke(
            [SystemMessage(content=_STRUCTURED_PROMPT), HumanMessage(content=human)]
        )
    except Exception as exc:
        logger.exception("Planner structured call failed")
        decision = PlannerDecision(verdict="HOLD", confidence="LOW",
                                   risk_notes=f"Planner error: {exc}")

    decision = _apply_hard_vetoes(decision, trigger, capital)
    tiers    = _kelly_tiers(sig_ctx, capital)
    plan     = None
    aid      = None

    if decision.verdict == "APPROVE":
        aid  = str(uuid.uuid4())[:8]
        plan = ExecutionPlan(
            venue=decision.venue, symbol=decision.symbol, side=decision.side,
            notional_usd=decision.notional_usd, order_type="market",
            reason=decision.reason,
            copy_trade_auto=tiers["auto"],
            copy_trade_show_low=tiers["show_low"],
            copy_trade_show_high=tiers["show_high"],
        )

    rec = _format_rec(decision, trigger, aid, tiers)
    logger.info("Planner v2: verdict=%s confidence=%s kelly_opt=%.2f",
                decision.verdict, decision.confidence, tiers["kelly_opt"])

    return {
        "verdict": decision.verdict,
        "confidence": decision.confidence,
        "execution_plan": plan,
        "risk_notes": decision.risk_notes,
        "formatted_recommendation": rec,
        "approval_id": aid,
        "phase": "awaiting_approval" if decision.verdict == "APPROVE" else "done",
    }
