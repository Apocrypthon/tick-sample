"""debate.py — adversarial debate with quant conviction scores."""
from __future__ import annotations

import asyncio
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from src.models import create_chat_model
from src.agents.trading.prompts import BULL_PROMPT, BEAR_PROMPT, DEBATE_PROMPT
from src.agents.trading.state import TradingState

logger = logging.getLogger(__name__)


def _format_reports(reports: list) -> str:
    return "\n\n".join(
        f"=== {r['analyst'].upper()} ANALYST (confidence={r['confidence']}) ===\n{r['content']}"
        for r in reports
    )


def _quant_conviction_block(sig_ctx: dict) -> str:
    """Summarise signal_context as a quant conviction header for the debate."""
    if not sig_ctx:
        return ""

    sigs   = sig_ctx.get("signals", {})
    regime = sig_ctx.get("regime", "UNKNOWN")
    fg     = sig_ctx.get("fear_greed")
    vix    = sig_ctx.get("vix", 0.0)

    if not sigs:
        return ""

    scores = [s.get("score", 0.0) for s in sigs.values()]
    avg    = sum(scores) / len(scores)
    bias   = "BULLISH" if avg > 0.15 else "BEARISH" if avg < -0.15 else "NEUTRAL"

    lines = [
        "=== QUANTITATIVE CONVICTION (pre-debate) ===",
        f"Composite signal score : {avg:+.4f}  →  {bias}",
        f"Regime                 : {regime}",
    ]
    if fg is not None:
        lines.append(f"Fear/Greed             : {fg}/100")
    if vix:
        lines.append(f"VIX proxy              : {vix:.1f}")

    for pid, s in sigs.items():
        lines.append(
            f"  {pid:<12} score={s.get('score', 0):+.3f}  "
            f"RSI={s.get('rsi', 0):.0f}  "
            f"vol={s.get('vol_regime','?')}  "
            f"kF5={s.get('kalman_forecast', 0):.2f}"
        )
    lines.append("=== END QUANT ===")
    return "\n".join(lines)


async def run_debate(state: TradingState, config: RunnableConfig) -> dict:
    """Adversarial bull/bear debate then synthesis into unified thesis."""
    reports  = state.get("analyst_reports", [])
    sig_ctx  = state.get("signal_context") or {}
    trigger  = state.get("trigger") or {}

    if not reports:
        logger.warning("Debate node: no reports — returning NEUTRAL")
        return {"debate_thesis": "NEUTRAL — no analyst data available.", "phase": "planning"}

    formatted      = _format_reports(reports)
    quant_block    = _quant_conviction_block(sig_ctx)
    analyst_context = (
        f"Trigger type: {trigger.get('type', 'unknown')}\n\n"
        f"{quant_block}\n\n"
        f"Analyst Reports:\n\n{formatted}"
    )

    model = create_chat_model(thinking_enabled=False)

    # ── phase 1: parallel bull/bear ──────────────────────────────────────
    bull_task = model.ainvoke([
        SystemMessage(content=BULL_PROMPT),
        HumanMessage(content=analyst_context),
    ])
    bear_task = model.ainvoke([
        SystemMessage(content=BEAR_PROMPT),
        HumanMessage(content=analyst_context),
    ])

    results = await asyncio.gather(bull_task, bear_task, return_exceptions=True)

    def _extract(r, label: str) -> str:
        if isinstance(r, Exception):
            logger.error("%s researcher failed: %s", label, r)
            return f"[{label.upper()} ERROR: {r}]"
        return r.content if isinstance(r.content, str) else str(r.content)

    bull_text = _extract(results[0], "bull")
    bear_text = _extract(results[1], "bear")
    logger.info("Debate: bull=%d chars, bear=%d chars", len(bull_text), len(bear_text))

    # ── phase 2: synthesis manager ───────────────────────────────────────
    synthesis_input = (
        f"Trigger type: {trigger.get('type', 'unknown')}\n\n"
        f"{quant_block}\n\n"
        f"=== BULL RESEARCHER ===\n{bull_text}\n\n"
        f"=== BEAR RESEARCHER ===\n{bear_text}\n\n"
        f"=== RAW ANALYST REPORTS ===\n{formatted}\n\n"
        "Synthesize into a unified trading thesis. Resolve conflicts per your rules."
    )

    try:
        resp   = await model.ainvoke([
            SystemMessage(content=DEBATE_PROMPT),
            HumanMessage(content=synthesis_input),
        ])
        thesis = resp.content if isinstance(resp.content, str) else str(resp.content)
        logger.info("Debate synthesis: %d chars", len(thesis))
    except Exception as exc:
        logger.exception("Debate synthesis failed")
        thesis = (
            f"[SYNTHESIS ERROR: {exc}]\n"
            f"Quant bias: {_quant_conviction_block(sig_ctx)}\n"
            f"Bull summary: {bull_text[:200]}\n"
            f"Bear summary: {bear_text[:200]}"
        )

    return {"debate_thesis": thesis, "phase": "planning"}
