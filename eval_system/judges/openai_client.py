"""Thin adapter over the OpenAI SDK's structured-output parsing. Not
unit-tested here for the same reason AnthropicJudgeClient isn't -- nothing
meaningful to assert without mocking the whole SDK; see judges/client.py for
the seam every metric actually depends on, and each judge metric's own tests
for the fake used in its place. Requires the `judge` extra
(`uv sync --extra judge`) and an OPENAI_API_KEY (see factory.py / README)."""
from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "gpt-4o"


class OpenAIJudgeClient:
    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.environ.get("OPENAI_JUDGE_MODEL", DEFAULT_MODEL)
        self._api_key = api_key
        self._client = None

    def _sdk_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as e:
                raise ImportError(
                    "OpenAIJudgeClient requires the `judge` extra: uv sync --extra judge"
                ) from e
            self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def structured_complete(self, prompt: str, response_model: type[T]) -> T:
        client = self._sdk_client()
        completion = client.beta.chat.completions.parse(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format=response_model,
        )
        return completion.choices[0].message.parsed
