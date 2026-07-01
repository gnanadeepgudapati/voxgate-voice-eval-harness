"""The one seam every judge metric depends on instead of an SDK directly, so
providers (Anthropic/OpenAI) or a test fake swap in without touching a metric."""
from __future__ import annotations

from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class JudgeClient(Protocol):
    def structured_complete(self, prompt: str, response_model: type[T]) -> T: ...
