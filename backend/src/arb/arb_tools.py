"""arb_tools.py — LangChain tools wrapping the ArbScanner + leg execution."""
from __future__ import annotations

import logging
import os
from typing import Optional, TYPE_CHECKING

from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from src.arb.arb_scanner import ArbScanner

logger = logging.getLogger(__name__)
_scanner: Optional["ArbScanner"] = None


def set_scanner(scanner) -> None:
    global _scanner
    _scanner = scanner


def _check_scanner() -> Optional[str]:
    if _scanner is None:
        return "ArbScanner not initialized."
    ready, total = _scanner.symbols_ready, _scanner.total_symbols
    if ready < total:
        return f"ArbScanner warming up ({ready}/{total} books). Retry in ~5s."
    return None


class ArbScanTool(BaseTool):
    name:        str = "arb_scan"
    description: str = (
        "Scan live Coinbase order books for triangular arbitrage "
        "across BTC/ETH/SOL. Returns best opportunity with profit "
        "percentage and leg prices. Read-only, safe to call anytime."
    )

    def _run(self, query: str = "") -> str:
        err = _check_scanner()
        if err:
            return err
        opp   = _scanner.get_latest()
        books = _scanner.get_book_summary()
        if opp is None:
            return f"No opportunity above {_scanner.capital_usd:.0f}% threshold.\nBooks: {books}"
        return f"{opp.format()}\n\nBooks: {books}"

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class ArbExecuteTool(BaseTool):
    name:        str = "arb_execute"
    description: str = (
        "Execute best triangular arb on Coinbase Advanced Trade. "
        "Fires 3 sequential market orders per leg. Capped at "
        "ARB_MAX_TRADE_USD. HARD GATE: TRADE_ENABLED=true required. "
        "Only call when arb_scan shows >= 0.30% profit AND the "
        "planner has issued an approval_id for this opportunity."
    )

    def _run(self, approval_id: str = "") -> str:
        if os.getenv("TRADE_ENABLED", "false").lower() != "true":
            return "TRADE_ENABLED=false — arb execution blocked (dry-run)."

        err = _check_scanner()
        if err:
            return err

        from src.arb.arb_scanner import EXEC_FLOOR_PCT
        opp = _scanner.get_latest()
        if opp is None:
            return "No current opportunity. Run arb_scan first."

        if opp.profit_pct < EXEC_FLOOR_PCT:
            return (
                f"HOLD: {opp.name} at {opp.profit_pct:.4f}% is below "
                f"{EXEC_FLOOR_PCT:.2f}% execution floor — slippage risk."
            )

        if not approval_id:
            return (
                "BLOCKED: no approval_id supplied. "
                "The planner must issue APPROVE with an approval_id "
                "before arb_execute can fire."
            )

        # ── execute each leg sequentially ─────────────────────────────────
        results = []
        capital = opp.capital_usd

        for sym, side, price in opp.legs:
            product_id = sym.replace("/", "-").upper()
            try:
                from src.tools.coinbase_tools import PlaceCryptoOrderTool
                result = PlaceCryptoOrderTool()._run(
                    product_id=product_id,
                    quote_size=round(capital, 2),
                    side=side.upper(),
                )
                results.append(f"  {side.upper()} {product_id}: {result}")
                logger.info(
                    "Arb leg executed: %s %s $%.2f  approval_id=%s",
                    side.upper(), product_id, capital, approval_id,
                )
            except Exception as exc:
                results.append(f"  {side.upper()} {product_id}: ERROR — {exc}")
                logger.error("Arb leg failed: %s %s — %s", side, product_id, exc)
                # Abort remaining legs on first failure to avoid orphaned positions
                results.append("  ABORTED: remaining legs skipped after leg failure")
                break

        summary = "\n".join(results)
        return (
            f"ARB EXECUTE: {opp.name}  profit_est={opp.profit_pct:+.4f}%\n"
            f"approval_id: {approval_id}\n"
            f"Capital: ${capital:.2f}\n"
            f"Legs:\n{summary}"
        )

    async def _arun(self, approval_id: str = "") -> str:
        return self._run(approval_id)


ARB_TOOLS = [ArbScanTool(), ArbExecuteTool()]
