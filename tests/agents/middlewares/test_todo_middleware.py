import pytest
import asyncio
from unittest.mock import MagicMock

# The dependencies (langchain, langgraph, etc.) are actually installed in the uv environment,
# so we don't need to mutate sys.modules at all!
from langchain_core.messages import AIMessage, HumanMessage
from src.agents.middlewares.todo_middleware import (
    _todos_in_messages,
    _reminder_in_messages,
    _format_todos,
    TodoMiddleware,
)

def test_todos_in_messages():
    """Test _todos_in_messages with various message lists."""
    # Empty messages
    assert _todos_in_messages([]) is False

    # AIMessage without tool calls
    assert _todos_in_messages([AIMessage(content="no tools")]) is False

    # AIMessage with other tool calls
    assert _todos_in_messages([AIMessage(content="", tool_calls=[{"name": "other_tool", "args": {}, "id": "1"}])]) is False

    # AIMessage with write_todos tool call
    assert _todos_in_messages([AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "1"}])]) is True

    # Mixed messages
    assert _todos_in_messages([
        HumanMessage(content="Hello"),
        AIMessage(content="", tool_calls=[{"name": "other_tool", "args": {}, "id": "1"}]),
        AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "2"}]),
    ]) is True

def test_reminder_in_messages():
    """Test _reminder_in_messages with various message lists."""
    # Empty messages
    assert _reminder_in_messages([]) is False

    # HumanMessage without name
    assert _reminder_in_messages([HumanMessage(content="No name")]) is False

    # HumanMessage with different name
    assert _reminder_in_messages([HumanMessage(content="", name="other")]) is False

    # HumanMessage with todo_reminder name
    assert _reminder_in_messages([HumanMessage(content="", name="todo_reminder")]) is True

    # AIMessage (even if it had a name attribute) should be ignored
    ai_msg = AIMessage(content="", name="todo_reminder")
    assert _reminder_in_messages([ai_msg]) is False

    # Mixed messages
    assert _reminder_in_messages([
        AIMessage(content=""),
        HumanMessage(content="", name="other"),
        HumanMessage(content="", name="todo_reminder"),
    ]) is True

def test_format_todos():
    """Test _format_todos function."""
    todos = [
        {"content": "Fix bug", "status": "done"},
        {"content": "Add feature", "status": "pending"},
        {"content": "Refactor code"}, # Missing status defaults to 'pending'
        {"status": "pending"}, # Missing content defaults to ''
    ]

    expected = (
        "- [done] Fix bug\n"
        "- [pending] Add feature\n"
        "- [pending] Refactor code\n"
        "- [pending] "
    )

    assert _format_todos(todos) == expected

def test_before_model_empty_todos():
    """Test before_model when todos list is empty or None."""
    mw = TodoMiddleware()
    runtime = MagicMock()

    # None todos
    assert mw.before_model({"todos": None}, runtime) is None

    # Empty todos
    assert mw.before_model({"todos": []}, runtime) is None

def test_before_model_todos_in_messages():
    """Test before_model when write_todos tool call is in messages."""
    mw = TodoMiddleware()
    runtime = MagicMock()

    state = {
        "todos": [{"content": "Task 1", "status": "pending"}],
        "messages": [AIMessage(content="", tool_calls=[{"name": "write_todos", "args": {}, "id": "1"}])]
    }

    assert mw.before_model(state, runtime) is None

def test_before_model_reminder_in_messages():
    """Test before_model when todo_reminder is already in messages."""
    mw = TodoMiddleware()
    runtime = MagicMock()

    state = {
        "todos": [{"content": "Task 1", "status": "pending"}],
        "messages": [HumanMessage(content="", name="todo_reminder")]
    }

    assert mw.before_model(state, runtime) is None

def test_before_model_inject_reminder():
    """Test before_model injecting reminder when context lost."""
    mw = TodoMiddleware()
    runtime = MagicMock()

    todos = [
        {"content": "Completed task", "status": "done"},
        {"content": "Pending task", "status": "pending"}
    ]

    state = {
        "todos": todos,
        "messages": [HumanMessage(content="User message")]
    }

    result = mw.before_model(state, runtime)

    assert result is not None
    assert "messages" in result
    assert len(result["messages"]) == 1

    reminder_msg = result["messages"][0]
    assert isinstance(reminder_msg, HumanMessage)
    assert reminder_msg.name == "todo_reminder"

    # Check content has formatted todos
    assert "<system_reminder>" in reminder_msg.content
    assert "- [done] Completed task" in reminder_msg.content
    assert "- [pending] Pending task" in reminder_msg.content

def test_abefore_model_delegates():
    """Test abefore_model delegates correctly to before_model."""
    mw = TodoMiddleware()
    runtime = MagicMock()
    state = {"todos": [{"content": "Task", "status": "pending"}]}

    # Since abefore_model simply calls before_model, we can just test if the results match
    sync_result = mw.before_model(state, runtime)
    async_result = asyncio.run(mw.abefore_model(state, runtime))

    assert sync_result == async_result
