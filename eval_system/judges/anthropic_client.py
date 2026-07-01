"""Thin adapter over the Anthropic SDK's structured-output tool-use pattern.
Not unit-tested here (no unit test meaningfully exercises a real network call
without mocking the SDK end to end -- see judges/client.py for the seam every
metric actually depends on and tests/test_faithfulness.py for the fake used in
its place). Requires the `judge` extra (`uv sync --extra judge`)."""
from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicJudgeClient:
    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        self.model = model
        self._api_key = api_key
        self._client = None

    def _sdk_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "AnthropicJudgeClient requires the `judge` extra: uv sync --extra judge"
                ) from e
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def structured_complete(self, prompt: str, response_model: type[T]) -> T:
        client = self._sdk_client()
        schema = response_model.model_json_schema()
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            tools=[{"name": "submit_judgment", "input_schema": schema}],
            tool_choice={"type": "tool", "name": "submit_judgment"},
            messages=[{"role": "user", "content": prompt}],
        )
        tool_use = next(block for block in response.content if block.type == "tool_use")
        return response_model.model_validate(tool_use.input)
