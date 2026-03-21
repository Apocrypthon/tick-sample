"""build_trading_graph — assembles the tick-switch multi-agent trading DAG.

Flow:
  START → hydrate (live data) → analysts (parallel 4x) → debate (bull/bear + synthesis) → planner → END

The graph produces a formatted recommendation in state['formatted_recommendation'].
Execution is HITL-gated: Discord reaction triggers run_execution() separately.
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from .nodes.hydrate import run_hydrate
from .nodes.analysts import run_analysts
from .nodes.debate import run_debate
from .nodes.planner import run_planner
from .state import TradingState

logger = logging.getLogger(__name__)


def build_trading_graph(checkpointer=None):
    g = StateGraph(TradingState)

    g.add_node("hydrate", run_hydrate)
    g.add_node("analysts", run_analysts)
    g.add_node("debate", run_debate)
    g.add_node("planner", run_planner)

    g.set_entry_point("hydrate")
    g.add_edge("hydrate", "analysts")
    g.add_edge("analysts", "debate")
    g.add_edge("debate", "planner")
    g.add_edge("planner", END)

    compiled = g.compile(checkpointer=checkpointer)
    logger.info("Trading DAG compiled: hydrate → analysts → debate → planner → END")
    return compiled


_graph = None


def get_trading_graph(checkpointer=None):
    """Module-level singleton — initialised lazily."""
    global _graph
    if _graph is None:
        _graph = build_trading_graph(checkpointer=checkpointer)
    return _graph
