"""Settings and API-key loading.

Values come from environment variables. A ``.env`` file in the working
directory (or any parent) is loaded automatically.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

DEFAULT_CHAT_MODEL = "gpt-5.2"
DEFAULT_TEMPERATURE = 0.2


class MissingAPIKeyError(RuntimeError):
    """Raised when a required API key is not set."""


def _first_env(*names: str) -> str | None:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the travel planner."""

    openai_api_key: str | None
    tavily_api_key: str | None
    serpapi_api_key: str | None
    chat_model: str = DEFAULT_CHAT_MODEL
    temperature: float = DEFAULT_TEMPERATURE

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from the environment (and ``.env`` if present)."""
        load_dotenv(find_dotenv(usecwd=True), override=False)

        serpapi_key = _first_env("SERPAPI_API_KEY", "SERP_API_KEY")
        if serpapi_key:
            # The flight/hotel tools read SERPAPI_API_KEY at call time.
            os.environ.setdefault("SERPAPI_API_KEY", serpapi_key)

        return cls(
            openai_api_key=_first_env("OPENAI_API_KEY"),
            tavily_api_key=_first_env("TAVILY_API_KEY"),
            serpapi_api_key=serpapi_key,
            chat_model=_first_env("OPENAI_CHAT_MODEL") or DEFAULT_CHAT_MODEL,
            temperature=float(_first_env("OPENAI_TEMPERATURE") or DEFAULT_TEMPERATURE),
        )

    def require_keys(self) -> None:
        """Raise :class:`MissingAPIKeyError` listing every missing key."""
        missing = [
            name
            for name, value in (
                ("OPENAI_API_KEY", self.openai_api_key),
                ("TAVILY_API_KEY", self.tavily_api_key),
                ("SERPAPI_API_KEY (or SERP_API_KEY)", self.serpapi_api_key),
            )
            if not value
        ]
        if missing:
            raise MissingAPIKeyError(
                "Missing API key(s): "
                + ", ".join(missing)
                + ". Set them in your environment or in a .env file (see .env.example)."
            )
