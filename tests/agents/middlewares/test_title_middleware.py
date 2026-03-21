import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from src.agents.middlewares.title_middleware import TitleMiddleware


# Mock config to make tests deterministic
class MockConfig:
    enabled = True
    max_words = 5
    max_chars = 40
    prompt_template = "Generate title for: User: {user_msg}, Assistant: {assistant_msg}"


@pytest.fixture
def mock_config():
    return MockConfig()


def test_generate_title_happy_path(mock_config):
    """Test the happy path where the model successfully generates a title."""
    middleware = TitleMiddleware()

    state = {
        "messages": [
            HumanMessage(content="Hello, what is 2+2?"),
            AIMessage(content="The answer is 4."),
        ]
    }

    # Setup the mock model and response
    mock_model = MagicMock()
    mock_response = MagicMock()
    # It strips out outer quotes, so testing it here
    mock_response.content = '"Simple Addition"'
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    # Patch the dependencies
    with (
        patch(
            "src.agents.middlewares.title_middleware.get_title_config",
            return_value=mock_config,
        ),
        patch(
            "src.agents.middlewares.title_middleware.create_chat_model",
            return_value=mock_model,
        ),
    ):
        # Call the actual method inside an asyncio event loop
        title = asyncio.run(middleware._generate_title(state))

        # Verify the title was stripped of quotes properly
        assert title == "Simple Addition"

        # Verify ainvoke was called with the correct prompt
        expected_prompt = mock_config.prompt_template.format(
            max_words=mock_config.max_words,
            user_msg="Hello, what is 2+2?",
            assistant_msg="The answer is 4.",
        )
        mock_model.ainvoke.assert_called_once_with(expected_prompt)


def test_generate_title_happy_path_long_title(mock_config):
    """Test the happy path where the generated title exceeds max_chars."""
    middleware = TitleMiddleware()

    state = {
        "messages": [
            HumanMessage(content="Can you tell me about the history of the universe?"),
            AIMessage(content="It all started with the Big Bang..."),
        ]
    }

    # Setup the mock model to return a very long title
    mock_model = MagicMock()
    mock_response = MagicMock()
    long_title = "This is a very long title that exceeds the maximum character limit set in the config"
    mock_response.content = long_title
    mock_model.ainvoke = AsyncMock(return_value=mock_response)

    # Patch the dependencies
    with (
        patch(
            "src.agents.middlewares.title_middleware.get_title_config",
            return_value=mock_config,
        ),
        patch(
            "src.agents.middlewares.title_middleware.create_chat_model",
            return_value=mock_model,
        ),
    ):
        # Call the method
        title = asyncio.run(middleware._generate_title(state))

        # Verify the title was truncated to max_chars
        assert title == long_title[: mock_config.max_chars]
        assert len(title) == mock_config.max_chars


def test_generate_title_fallback_long_user_message(mock_config):
    """Test the fallback path when model.ainvoke throws an exception, with a long user message."""
    middleware = TitleMiddleware()

    long_user_msg = "This is a really really long message that exceeds the fallback character limit."
    state = {
        "messages": [
            HumanMessage(content=long_user_msg),
            AIMessage(content="Yes, it is."),
        ]
    }

    # Setup the mock model to throw an exception
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=Exception("API limit reached"))

    # Patch the dependencies
    with (
        patch(
            "src.agents.middlewares.title_middleware.get_title_config",
            return_value=mock_config,
        ),
        patch(
            "src.agents.middlewares.title_middleware.create_chat_model",
            return_value=mock_model,
        ),
    ):
        # Call the method
        title = asyncio.run(middleware._generate_title(state))

        # Verify fallback logic: uses max_chars or 50, whichever is smaller.
        fallback_chars = min(mock_config.max_chars, 50)
        assert title == long_user_msg[:fallback_chars].rstrip() + "..."


def test_generate_title_fallback_short_user_message(mock_config):
    """Test the fallback path when model.ainvoke throws an exception, with a short user message."""
    middleware = TitleMiddleware()

    short_user_msg = "Short message"
    state = {
        "messages": [HumanMessage(content=short_user_msg), AIMessage(content="Ok.")]
    }

    # Setup the mock model to throw an exception
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=Exception("Timeout"))

    # Patch the dependencies
    with (
        patch(
            "src.agents.middlewares.title_middleware.get_title_config",
            return_value=mock_config,
        ),
        patch(
            "src.agents.middlewares.title_middleware.create_chat_model",
            return_value=mock_model,
        ),
    ):
        # Call the method
        title = asyncio.run(middleware._generate_title(state))

        # Verify fallback logic for short string
        assert title == short_user_msg


def test_generate_title_fallback_empty_user_message(mock_config):
    """Test the fallback path when model.ainvoke throws an exception, with an empty user message."""
    middleware = TitleMiddleware()

    state = {"messages": [HumanMessage(content=""), AIMessage(content="Hello there")]}

    # Setup the mock model to throw an exception
    mock_model = MagicMock()
    mock_model.ainvoke = AsyncMock(side_effect=Exception("Another Error"))

    # Patch the dependencies
    with (
        patch(
            "src.agents.middlewares.title_middleware.get_title_config",
            return_value=mock_config,
        ),
        patch(
            "src.agents.middlewares.title_middleware.create_chat_model",
            return_value=mock_model,
        ),
    ):
        # Call the method
        title = asyncio.run(middleware._generate_title(state))

        # Verify ultimate fallback
        assert title == "New Conversation"
