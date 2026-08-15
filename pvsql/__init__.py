"""PV-SQL: Probe-and-Verify text-to-SQL.

The framework is model-agnostic. `PVSQL` and `SQLiteEnv` import nothing but the
standard library -- pass any `(messages, temperature) -> str` callable as `llm`.

The names re-exported from `pvsql.llm` (`chat_completion`, `ConfigError`,
`get_token_usage`, `reset_token_usage`) are a convenience adapter for
OpenAI-compatible endpoints. They are resolved lazily, so `openai` and
`tenacity` are only needed if you actually use them.
"""

from typing import TYPE_CHECKING

from .db import DatabaseEnv, SQLiteEnv
from .pv_sql import PVSQL, LLMFn, PVSQLResult, generate_sql

__version__ = "0.1.0"

_LLM_EXPORTS = {
    "chat_completion",
    "ConfigError",
    "get_token_usage",
    "reset_token_usage",
    "reset_client",
}


def __getattr__(name: str):
    """Resolve the optional LLM adapter on first access (PEP 562)."""
    if name in _LLM_EXPORTS:
        from . import llm

        return getattr(llm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)


if TYPE_CHECKING:  # for type checkers and IDE completion only
    from .llm import (  # noqa: F401
        ConfigError,
        chat_completion,
        get_token_usage,
        reset_client,
        reset_token_usage,
    )

__all__ = [
    # framework -- stdlib only
    "PVSQL",
    "PVSQLResult",
    "generate_sql",
    "LLMFn",
    "DatabaseEnv",
    "SQLiteEnv",
    # optional OpenAI-compatible adapter
    "chat_completion",
    "ConfigError",
    "get_token_usage",
    "reset_token_usage",
    "reset_client",
]
