from anthropic import APIError
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch

import main
import pytest

http_client = TestClient(main.app)


# --- Fixtures ---

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the cache before and after every test so tests don't bleed into each other."""
    main.cache.clear()
    yield


# --- Helpers ---

def make_mock_response(text="Hello!", input_tokens=10, output_tokens=5):
    """Fake Anthropic response for client.messages.create()"""
    mock = MagicMock()
    mock.content[0].text = text
    mock.usage.input_tokens = input_tokens
    mock.usage.output_tokens = output_tokens
    return mock


def make_mock_stream(text_chunks=None):
    """Fake Anthropic stream for client.messages.stream()"""
    if text_chunks is None:
        text_chunks = ["Hello", "!"]
    mock_stream = MagicMock()
    mock_stream.__enter__ = MagicMock(return_value=mock_stream)
    mock_stream.__exit__ = MagicMock(return_value=False)
    mock_stream.text_stream = iter(text_chunks)

    final_message = MagicMock()
    final_message.content[0].text = "".join(text_chunks)
    final_message.usage.input_tokens = 10
    final_message.usage.output_tokens = 5
    mock_stream.get_final_message.return_value = final_message

    return mock_stream


# --- /health ---

def test_health():
    response = http_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /models ---

def test_list_models():
    response = http_client.get("/models")
    assert response.status_code == 200
    assert "haiku" in response.json()["models"]
    assert "sonnet" in response.json()["models"]


# --- /chat ---

def test_chat_returns_expected_fields():
    with patch.object(main.client.messages, "create", return_value=make_mock_response()) as mock_create:
        response = http_client.post("/chat", json={"message": "hello"})
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Hello!"
        assert data["cached"] == False
        assert data["model"] == "haiku"
        assert "usage" in data
        assert "estimated_cost_usd" in data
        mock_create.assert_called_once()


def test_chat_invalid_model_returns_error():
    response = http_client.post("/chat", json={"message": "hello", "model": "gpt-4"})
    assert response.status_code == 400


def test_chat_second_request_returns_from_cache():
    with patch.object(main.client.messages, "create", return_value=make_mock_response()) as mock_create:
        http_client.post("/chat", json={"message": "hello"})
        response = http_client.post("/chat", json={"message": "hello"})
        assert response.json()["cached"] == True
        assert mock_create.call_count == 1  # Anthropic was only called once


def test_chat_different_models_cached_separately():
    with patch.object(main.client.messages, "create", return_value=make_mock_response()) as mock_create:
        http_client.post("/chat", json={"message": "hello", "model": "haiku"})
        http_client.post("/chat", json={"message": "hello", "model": "sonnet"})
        assert mock_create.call_count == 2  # different models = different cache keys


def test_chat_api_error_returns_503():
    mock_error = APIError("upstream error", request=MagicMock(), body=None)
    with patch.object(main.client.messages, "create", side_effect=mock_error):
        response = http_client.post("/chat", json={"message": "hello"})
        assert response.status_code == 503


def test_chat_cost_calculation():
    with patch.object(main.client.messages, "create", return_value=make_mock_response(input_tokens=1_000_000, output_tokens=1_000_000)):
        response = http_client.post("/chat", json={"message": "hello"})
        # haiku: $0.80/M input + $4.00/M output = $4.80 for 1M of each
        assert response.json()["estimated_cost_usd"] == 4.80


# --- /chat/stream ---

def test_chat_stream_returns_text():
    with patch.object(main.client.messages, "stream", return_value=make_mock_stream(["Hello", "!"])):
        response = http_client.post("/chat/stream", json={"message": "hello"})
        assert response.status_code == 200
        assert "Hello!" in response.text


def test_chat_stream_cache_hit_returns_header():
    with patch.object(main.client.messages, "stream", return_value=make_mock_stream()) as mock_stream:
        http_client.post("/chat/stream", json={"message": "hello"})
        response = http_client.post("/chat/stream", json={"message": "hello"})
        assert response.headers["x-cache"] == "HIT"
        assert mock_stream.call_count == 1
