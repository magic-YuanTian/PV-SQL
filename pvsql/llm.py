"""Optional OpenAI-compatible adapter for PV-SQL.

This module is a convenience, not a dependency. The framework takes the model
as a plain `(messages, temperature) -> str` callable, so you can ignore
everything here and pass your own:

    PVSQL("db.sqlite", llm=my_callable)

Credentials are read from environment variables only -- nothing is hardcoded.
Copy `.env.example` to `.env` and fill in your own values, or export them in
your shell.

Two providers are supported:

  PVSQL_LLM_PROVIDER=openai   (default)
      OPENAI_API_KEY      required
      OPENAI_BASE_URL     optional -- any OpenAI-compatible endpoint
      PVSQL_MODEL         model name, default "gpt-4o"

  PVSQL_LLM_PROVIDER=azure
      AZURE_OPENAI_API_KEY      required
      AZURE_OPENAI_ENDPOINT     required, e.g. https://<resource>.openai.azure.com/
      AZURE_OPENAI_API_VERSION  optional, default "2024-10-01-preview"
      PVSQL_MODEL               Azure *deployment* name, default "gpt-4o"
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


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(
            f"Environment variable {name} is not set. "
            f"Copy .env.example to .env and fill it in, or export {name} in your shell."
        )
    return value


def _build_client():
    provider = os.getenv("PVSQL_LLM_PROVIDER", "openai").strip().lower()

    if provider == "azure":
        from openai import AzureOpenAI

        return AzureOpenAI(
            azure_endpoint=_require("AZURE_OPENAI_ENDPOINT"),
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
            timeout=DEFAULT_TIMEOUT,
        )

    if provider == "openai":
        from openai import OpenAI

        kwargs = {"api_key": _require("OPENAI_API_KEY"), "timeout": DEFAULT_TIMEOUT}
        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    raise ConfigError(
        f"Unknown PVSQL_LLM_PROVIDER={provider!r}. Expected 'openai' or 'azure'."
    )


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
        _thread_local.tokens = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
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
