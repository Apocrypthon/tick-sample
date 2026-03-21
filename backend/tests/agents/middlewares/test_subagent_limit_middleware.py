import sys
import asyncio
from unittest.mock import MagicMock, patch

# In a restricted environment with missing dependencies, we mock the required
# modules to allow importing the middleware and running unit tests.
MOCK_MODULES = [
    "langchain",
    "langchain.agents",
    "langchain.agents.middleware",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.runnables",
    "langgraph",
    "langgraph.runtime",
    "langgraph.types",
    "yaml",
    "pydantic",
    "dotenv",
    "src.agents.checkpointer",
    "src.agents.lead_agent",
    "src.subagents",
    "src.subagents.executor",
]

# Apply mocks to sys.modules BEFORE any imports
for mod in MOCK_MODULES:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()

# Configure specific mock behavior needed for the middleware's base class
class MockBase:
    def __class_getitem__(cls, item):
        return cls
    def __init__(self, *args, **kwargs):
        pass

sys.modules["langchain.agents.middleware"].AgentMiddleware = MockBase
sys.modules["langchain.agents"].AgentState = dict

# Mock the internal import from src.subagents.executor
executor_mock = MagicMock()
executor_mock.MAX_CONCURRENT_SUBAGENTS = 3
sys.modules["src.subagents.executor"] = executor_mock

# Now we can safely import the actual middleware code
from src.agents.middlewares.subagent_limit_middleware import (
    _clamp_subagent_limit,
    SubagentLimitMiddleware,
    MIN_SUBAGENT_LIMIT,
    MAX_SUBAGENT_LIMIT,
)

def test_clamp_subagent_limit():
    """Verify that subagent limit is clamped within [MIN, MAX] range."""
    assert _clamp_subagent_limit(1) == MIN_SUBAGENT_LIMIT
    assert _clamp_subagent_limit(2) == 2
    assert _clamp_subagent_limit(3) == 3
    assert _clamp_subagent_limit(4) == 4
    assert _clamp_subagent_limit(5) == MAX_SUBAGENT_LIMIT

def test_middleware_init_clamping():
    """Verify that SubagentLimitMiddleware clamps its max_concurrent parameter."""
    mw = SubagentLimitMiddleware(max_concurrent=1)
    assert mw.max_concurrent == MIN_SUBAGENT_LIMIT

    mw = SubagentLimitMiddleware(max_concurrent=3)
    assert mw.max_concurrent == 3

    mw = SubagentLimitMiddleware(max_concurrent=5)
    assert mw.max_concurrent == MAX_SUBAGENT_LIMIT

def test_truncate_task_calls_no_messages():
    """Verify handling of empty state or missing messages."""
    mw = SubagentLimitMiddleware()
    assert mw._truncate_task_calls({}) is None
    assert mw._truncate_task_calls({"messages": []}) is None

def test_truncate_task_calls_not_ai_message():
    """Verify that only AI messages are processed for truncation."""
    mw = SubagentLimitMiddleware()
    msg = MagicMock()
    msg.type = "human"
    assert mw._truncate_task_calls({"messages": [msg]}) is None

def test_truncate_task_calls_no_tool_calls():
    """Verify handling of AI messages without any tool calls."""
    mw = SubagentLimitMiddleware()
    msg = MagicMock()
    msg.type = "ai"
    msg.tool_calls = None
    assert mw._truncate_task_calls({"messages": [msg]}) is None

def test_truncate_task_calls_within_limit():
    """Verify that 'task' tool calls within the limit are not truncated."""
    mw = SubagentLimitMiddleware(max_concurrent=2)
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [{"name": "task"}, {"name": "task"}]
    assert mw._truncate_task_calls({"messages": [ai_msg]}) is None

def test_truncate_task_calls_exceed_limit():
    """Verify that excess 'task' tool calls are truncated."""
    mw = SubagentLimitMiddleware(max_concurrent=2)
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [
        {"name": "task", "id": "1"},
        {"name": "task", "id": "2"},
        {"name": "task", "id": "3"}
    ]

    # Mock model_copy to simulate creating a new message with updated tool_calls
    ai_msg.model_copy.side_effect = lambda update: MagicMock(tool_calls=update["tool_calls"])

    result = mw._truncate_task_calls({"messages": [ai_msg]})
    assert result is not None
    updated_msg = result["messages"][0]
    assert len(updated_msg.tool_calls) == 2
    assert updated_msg.tool_calls == [{"name": "task", "id": "1"}, {"name": "task", "id": "2"}]

def test_truncate_task_calls_mixed_tools():
    """Verify that only excess 'task' calls are truncated, preserving other tools."""
    # max_concurrent=2 for this test
    mw = SubagentLimitMiddleware(max_concurrent=2)
    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [
        {"name": "get_quote", "id": "1"},
        {"name": "task", "id": "2"},
        {"name": "task", "id": "3"},
        {"name": "task", "id": "4"},
        {"name": "other", "id": "5"}
    ]

    ai_msg.model_copy.side_effect = lambda update: MagicMock(tool_calls=update["tool_calls"])

    result = mw._truncate_task_calls({"messages": [ai_msg]})
    assert result is not None
    updated_msg = result["messages"][0]
    # Expected: 2 tasks kept (ids: 2, 3), other tools preserved (ids: 1, 5)
    expected_tools = [
        {"name": "get_quote", "id": "1"},
        {"name": "task", "id": "2"},
        {"name": "task", "id": "3"},
        {"name": "other", "id": "5"}
    ]
    assert updated_msg.tool_calls == expected_tools

def test_middleware_hooks_delegation():
    """Verify that after_model and aafter_model delegate to _truncate_task_calls."""
    mw = SubagentLimitMiddleware()
    state = {"messages": []}
    runtime = MagicMock()

    with patch.object(mw, "_truncate_task_calls", return_value={"messages": ["updated"]}) as mock_truncate:
        # Sync hook
        result_sync = mw.after_model(state, runtime)
        mock_truncate.assert_called_with(state)
        assert result_sync == {"messages": ["updated"]}

        # Async hook
        result_async = asyncio.run(mw.aafter_model(state, runtime))
        mock_truncate.assert_called_with(state)
        assert result_async == {"messages": ["updated"]}
