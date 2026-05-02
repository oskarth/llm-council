import importlib
import os
import unittest
from unittest.mock import patch

from backend import openrouter


class ConfigTests(unittest.TestCase):
    def test_env_overrides_and_defaults(self):
        env = {
            "COUNCIL_MODELS": "openai/gpt-5.1, anthropic/claude-sonnet-4.5",
            "CHAIRMAN_MODEL": "x-ai/grok-4",
            "TITLE_MODEL": "google/gemini-2.5-flash",
            "DATA_DIR": "tmp/conversations",
        }

        with patch.dict(os.environ, env, clear=False):
            from backend import config

            importlib.reload(config)

            self.assertEqual(
                config.COUNCIL_MODELS,
                ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"],
            )
            self.assertEqual(config.CHAIRMAN_MODEL, "x-ai/grok-4")
            self.assertEqual(config.TITLE_MODEL, "google/gemini-2.5-flash")
            self.assertEqual(config.DATA_DIR, "tmp/conversations")

        from backend import config

        importlib.reload(config)

    def test_default_models_are_current(self):
        from backend import config

        self.assertIn("openai/gpt-5.1-chat", config.DEFAULT_COUNCIL_MODELS)
        self.assertIn("google/gemini-3.1-pro-preview", config.DEFAULT_COUNCIL_MODELS)
        self.assertEqual(config.CHAIRMAN_MODEL, os.getenv("CHAIRMAN_MODEL", "openai/gpt-5.1"))


class NormalizeContentTests(unittest.TestCase):
    def test_normalizes_string_content(self):
        self.assertEqual(openrouter._normalize_content("hello"), "hello")

    def test_normalizes_block_content(self):
        content = [
            {"type": "text", "text": "first"},
            {"type": "output_text", "text": "second"},
            {"content": "third"},
        ]

        self.assertEqual(openrouter._normalize_content(content), "first\nsecond\nthird")


class QueryModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_query_model_returns_none_without_api_key(self):
        with patch.object(openrouter, "OPENROUTER_API_KEY", None):
            response = await openrouter.query_model(
                "openai/gpt-5.1-chat",
                [{"role": "user", "content": "hi"}],
            )

        self.assertIsNone(response)

    async def test_query_model_normalizes_non_string_response_content(self):
        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {"type": "text", "text": "hello"},
                                    {"type": "output_text", "text": "world"},
                                ],
                                "reasoning_details": [{"type": "summary"}],
                            }
                        }
                    ]
                }

        class FakeAsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def post(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(openrouter, "OPENROUTER_API_KEY", "test-key"):
            with patch.object(openrouter.httpx, "AsyncClient", FakeAsyncClient):
                response = await openrouter.query_model(
                    "openai/gpt-5.1-chat",
                    [{"role": "user", "content": "hi"}],
                )

        self.assertEqual(response["content"], "hello\nworld")
        self.assertEqual(response["reasoning_details"], [{"type": "summary"}])


if __name__ == "__main__":
    unittest.main()
