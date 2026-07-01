"""Picks the default judge client for metrics that don't have one injected
(the registry-instantiated singletons, e.g. FaithfulnessMetric() with no
`client` arg). Controlled by VOXGATE_JUDGE_PROVIDER (anthropic|openai),
defaulting to anthropic per CLAUDE.md's tech list -- lets a user switch
providers with one env var instead of editing every judge metric."""
from __future__ import annotations

import os

from eval_system.judges.client import JudgeClient


def get_default_judge_client() -> JudgeClient:
    provider = os.environ.get("VOXGATE_JUDGE_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from eval_system.judges.anthropic_client import AnthropicJudgeClient

        return AnthropicJudgeClient()
    if provider == "openai":
        from eval_system.judges.openai_client import OpenAIJudgeClient

        return OpenAIJudgeClient()

    raise ValueError(f"Unknown VOXGATE_JUDGE_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')")
