"""Picks the default judge client for metrics that don't have one injected
(the registry-instantiated singletons, e.g. FaithfulnessMetric() with no
`client` arg). Controlled by VOXGATE_JUDGE_PROVIDER (anthropic|openai),
defaulting to anthropic per CLAUDE.md's tech list -- lets a user switch
providers with one env var instead of editing every judge metric.

Also loads `.env` at the repo root (API keys go there, see .env.example) so
neither provider's key needs to be set in the shell every session -- an
already-set shell/OS environment variable always wins over the file."""
from __future__ import annotations

import os
from pathlib import Path

from eval_system.judges.client import JudgeClient

REPO_ROOT_DOTENV = Path(__file__).resolve().parents[2] / ".env"


def load_dotenv_if_available(dotenv_path: Path | None = None) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(dotenv_path if dotenv_path is not None else REPO_ROOT_DOTENV)


def get_default_judge_client() -> JudgeClient:
    # Reference the module attribute at call time (not as a default parameter
    # value, which would bind once at import time) so tests can monkeypatch
    # REPO_ROOT_DOTENV to avoid depending on a developer's real, gitignored .env.
    load_dotenv_if_available(REPO_ROOT_DOTENV)
    provider = os.environ.get("VOXGATE_JUDGE_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        from eval_system.judges.anthropic_client import AnthropicJudgeClient

        return AnthropicJudgeClient()
    if provider == "openai":
        from eval_system.judges.openai_client import OpenAIJudgeClient

        return OpenAIJudgeClient()

    raise ValueError(f"Unknown VOXGATE_JUDGE_PROVIDER: {provider!r} (expected 'anthropic' or 'openai')")
