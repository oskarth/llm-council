"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()


def _parse_model_list(raw_value: str | None, default: list[str]) -> list[str]:
    """Parse a comma-separated model list from the environment."""
    if raw_value is None:
        return default

    models = [model.strip() for model in raw_value.split(",") if model.strip()]
    return models or default


# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers.
# These defaults are current on OpenRouter as of May 3, 2026.
DEFAULT_COUNCIL_MODELS = [
    "openai/gpt-5.1-chat",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]
COUNCIL_MODELS = _parse_model_list(
    os.getenv("COUNCIL_MODELS"),
    DEFAULT_COUNCIL_MODELS,
)

# Chairman and title models.
CHAIRMAN_MODEL = os.getenv("CHAIRMAN_MODEL", "openai/gpt-5.1")
TITLE_MODEL = os.getenv("TITLE_MODEL", "google/gemini-2.5-flash")

# OpenRouter API endpoint
OPENROUTER_API_URL = os.getenv(
    "OPENROUTER_API_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)

# Optional OpenRouter attribution headers
OPENROUTER_APP_NAME = os.getenv("OPENROUTER_APP_NAME", "llm-council")
OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL")

# Data directory for conversation storage
DATA_DIR = os.getenv("DATA_DIR", "data/conversations")
