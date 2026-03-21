import logging, os
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
    name: str = "arb_scan"
    description: str = (
        "Scan live Coinbase order books for triangular arbitrage across BTC/ETH/SOL. "
        "Returns best opportunity with profit % and leg prices. Read-only."
    )

    def _run(self, query: str = "") -> str:
        err = _check_scanner()
        if err:
            return err
        opp = _scanner.get_latest()
        books = _scanner.get_book_summary()
        if opp is None:
            return f"No opportunity above 0.05% threshold.\nBooks: {books}"
        return f"{opp.format()}\n\nBooks: {books}"

    async def _arun(self, query: str = "") -> str:
        return self._run(query)


class ArbExecuteTool(BaseTool):
    name: str = "arb_execute"
    description: str = (
        "Execute best triangular arb on Coinbase Advanced Trade. "
        "3 sequential market orders. Capped at ARB_MAX_TRADE_USD. "
        "HARD GATE: TRADE_ENABLED=true required. Only call when arb_scan shows >= 0.30% profit."
    )

    def _run(self, confirm: str = "") -> str:
        if os.getenv("TRADE_ENABLED", "false").lower() != "true":
            return "TRADE_ENABLED=false — arb execution blocked."
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
                f"{EXEC_FLOOR_PCT:.2f}% execution floor. Slippage risk."
            )
        # Execution stub — wire to place_crypto_order per leg to activate
        return (
            f"ARB EXECUTE (STUB)\n"
            f"Triangle: {opp.name}\n"
            f"Capital: ${opp.capital_usd:.2f} | Est profit: ${opp.gross_profit_usd:.4f}\n"
            f"Leg engine not yet wired. Safe no-op."
        )

    async def _arun(self, confirm: str = "") -> str:
        return self._run(confirm)


ARB_TOOLS = [ArbScanTool(), ArbExecuteTool()]
