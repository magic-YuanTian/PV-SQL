"""Optional LLM adapter for PV-SQL.

This module is a convenience, not a dependency. PV-SQL takes the model as a
plain `(messages, temperature) -> str` callable, so you can ignore everything
here and pass your own:

    PVSQL("db.sqlite", llm=my_callable)

What follows is a minimal adapter for chat-completions endpoints, provided so
the examples run out of the box. Configure it with three environment
variables -- nothing is hardcoded:

    PVSQL_API_KEY    required
    PVSQL_MODEL      model name, default "gpt-4o"
    PVSQL_BASE_URL   optional, for a non-default endpoint
"""

from __future__ import annotations

import os
import threading
from typing import Dict, List, Optional

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)


class ConfigError(RuntimeError):
    """Bad or missing configuration. Never worth retrying."""


DEFAULT_MODEL = os.getenv("PVSQL_MODEL", "gpt-4o")
DEFAULT_TIMEOUT = float(os.getenv("PVSQL_LLM_TIMEOUT", "180"))

_client = None
_client_lock = threading.Lock()
_thread_local = threading.local()


def _load_dotenv_if_present() -> None:
    """Load a local .env file when python-dotenv is installed. Optional."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


_load_dotenv_if_present()


def _build_client():
    api_key = os.getenv("PVSQL_API_KEY")
    if not api_key:
        raise ConfigError(
            "PVSQL_API_KEY is not set. Copy .env.example to .env and fill it in, "
            "export PVSQL_API_KEY in your shell, or pass your own `llm` callable "
            "to PVSQL(...) and ignore this module."
        )

    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ConfigError(
            "The bundled adapter needs the `openai` package: pip install openai. "
            "Alternatively pass your own `llm` callable to PVSQL(...)."
        ) from e

    kwargs = {"api_key": api_key, "timeout": DEFAULT_TIMEOUT}
    base_url = os.getenv("PVSQL_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def get_client():
    """Return the process-wide client, building it on first use."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = _build_client()
    return _client


def reset_client() -> None:
    """Drop the cached client so the next call re-reads the environment."""
    global _client
    with _client_lock:
        _client = None


# --- token accounting -------------------------------------------------------
# Tracked per-thread so a thread-pool over many questions gives per-question
# numbers rather than one shared total.


def _tracker() -> Dict[str, int]:
    if not hasattr(_thread_local, "tokens"):
        reset_token_usage()
    return _thread_local.tokens


def reset_token_usage() -> None:
    _thread_local.tokens = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }


def get_token_usage() -> Dict[str, int]:
    return dict(_tracker())


@retry(
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    retry=retry_if_not_exception_type(ConfigError),
    reraise=True,
)
def chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0,
    model: Optional[str] = None,
) -> str:
    """Send a chat request and return the assistant message content.

    Transient API errors are retried with exponential backoff; a `ConfigError`
    is raised immediately so a missing key fails in a second rather than after
    six backoff rounds.
    """
    completion = get_client().chat.completions.create(
        model=model or DEFAULT_MODEL,
        messages=messages,
        temperature=temperature,
    )

    usage = getattr(completion, "usage", None)
    if usage:
        t = _tracker()
        t["prompt_tokens"] += usage.prompt_tokens or 0
        t["completion_tokens"] += usage.completion_tokens or 0
        t["total_tokens"] += usage.total_tokens or 0

    return completion.choices[0].message.content


if __name__ == "__main__":
    print(chat_completion([{"role": "user", "content": "Reply with the word: ok"}]))
    print(get_token_usage())
