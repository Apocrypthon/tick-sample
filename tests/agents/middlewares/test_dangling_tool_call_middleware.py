import sys
from unittest.mock import MagicMock

# Apply mocks to sys.modules BEFORE any imports
MOCK_MODULES = [
    "langchain",
    "langchain.agents",
    "langchain.agents.middleware",
    "langchain.agents.middleware.types",
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.runnables",
    "langgraph",
    "langgraph.types",
    "langgraph.runtime",
    "yaml",
    "pydantic",
    "dotenv",
    "src.agents.checkpointer",
    "src.agents.lead_agent",
    "src.subagents",
    "src.subagents.executor",
]

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

from unittest.mock import patch

# Now we can safely import the actual middleware code
from src.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware

# We create a local MockToolMessage for testing without messing up sys.modules.
class MockToolMessage:
    def __init__(self, content, tool_call_id, name="unknown", status="success"):
        self.content = content
        self.tool_call_id = tool_call_id
        self.name = name
        self.status = status
        self.type = "tool"

    def __eq__(self, other):
        if not isinstance(other, MockToolMessage):
            return False
        return (self.content == other.content and
                self.tool_call_id == other.tool_call_id and
                self.name == other.name and
                self.status == other.status)


@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_no_messages():
    """Verify handling of empty message list."""
    mw = DanglingToolCallMiddleware()
    assert mw._build_patched_messages([]) is None

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_no_ai_message():
    """Verify handling of messages without AI messages."""
    mw = DanglingToolCallMiddleware()
    msg = MagicMock()
    msg.type = "human"
    assert mw._build_patched_messages([msg]) is None

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_ai_no_tool_calls():
    """Verify handling of AI message with no tool calls."""
    mw = DanglingToolCallMiddleware()
    msg = MagicMock()
    msg.type = "ai"
    msg.tool_calls = None
    assert mw._build_patched_messages([msg]) is None

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_all_matched():
    """Verify that if all tool calls have a corresponding ToolMessage, no patching is done."""
    mw = DanglingToolCallMiddleware()

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [{"name": "task", "id": "tc1"}]

    tool_msg = MockToolMessage(content="Success", tool_call_id="tc1", name="task")

    assert mw._build_patched_messages([ai_msg, tool_msg]) is None

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_dangling_tool_call():
    """Verify that a dangling tool call gets patched."""
    mw = DanglingToolCallMiddleware()

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [{"name": "task", "id": "tc1"}]

    result = mw._build_patched_messages([ai_msg])

    assert result is not None
    assert len(result) == 2
    assert result[0] == ai_msg
    assert isinstance(result[1], MockToolMessage)
    assert result[1].tool_call_id == "tc1"
    assert result[1].status == "error"
    assert result[1].content == "[Tool call was interrupted and did not return a result.]"

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_multiple_dangling():
    """Verify patching of multiple dangling tool calls on a single AIMessage."""
    mw = DanglingToolCallMiddleware()

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [
        {"name": "task1", "id": "tc1"},
        {"name": "task2", "id": "tc2"}
    ]

    result = mw._build_patched_messages([ai_msg])

    assert result is not None
    assert len(result) == 3
    assert result[0] == ai_msg
    assert isinstance(result[1], MockToolMessage)
    assert result[1].tool_call_id == "tc1"
    assert result[1].name == "task1"
    assert isinstance(result[2], MockToolMessage)
    assert result[2].tool_call_id == "tc2"
    assert result[2].name == "task2"

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_mixed_matched_and_dangling():
    """Verify that we only patch dangling calls, not matched ones."""
    mw = DanglingToolCallMiddleware()

    ai_msg = MagicMock()
    ai_msg.type = "ai"
    ai_msg.tool_calls = [
        {"name": "task1", "id": "tc1"},
        {"name": "task2", "id": "tc2"}
    ]

    tool_msg = MockToolMessage(content="Success", tool_call_id="tc1", name="task1")

    result = mw._build_patched_messages([ai_msg, tool_msg])

    assert result is not None
    # Original list is 2 messages (ai + tool). We add 1 patched tool message after ai.
    # Order: [ai_msg, patched_tool_msg, tool_msg]
    assert len(result) == 3
    assert result[0] == ai_msg
    assert isinstance(result[1], MockToolMessage)
    assert result[1].tool_call_id == "tc2"
    assert result[1].status == "error"
    assert result[2] == tool_msg

@patch("src.agents.middlewares.dangling_tool_call_middleware.ToolMessage", MockToolMessage)
def test_build_patched_messages_multiple_ai_messages():
    """Verify that multiple AI messages get patched at the correct positions."""
    mw = DanglingToolCallMiddleware()

    ai_msg1 = MagicMock()
    ai_msg1.type = "ai"
    ai_msg1.tool_calls = [{"name": "task1", "id": "tc1"}]

    human_msg = MagicMock()
    human_msg.type = "human"

    ai_msg2 = MagicMock()
    ai_msg2.type = "ai"
    ai_msg2.tool_calls = [{"name": "task2", "id": "tc2"}]

    result = mw._build_patched_messages([ai_msg1, human_msg, ai_msg2])

    assert result is not None
    assert len(result) == 5
    assert result[0] == ai_msg1
    assert result[1].tool_call_id == "tc1"
    assert result[2] == human_msg
    assert result[3] == ai_msg2
    assert result[4].tool_call_id == "tc2"
