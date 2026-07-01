import pytest

from eval_system.judges.factory import get_default_judge_client


def test_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("VOXGATE_JUDGE_PROVIDER", raising=False)

    client = get_default_judge_client()

    assert type(client).__name__ == "AnthropicJudgeClient"


def test_selects_openai_via_env_var(monkeypatch):
    monkeypatch.setenv("VOXGATE_JUDGE_PROVIDER", "openai")

    client = get_default_judge_client()

    assert type(client).__name__ == "OpenAIJudgeClient"


def test_provider_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("VOXGATE_JUDGE_PROVIDER", "OpenAI")

    client = get_default_judge_client()

    assert type(client).__name__ == "OpenAIJudgeClient"


def test_unknown_provider_raises(monkeypatch):
    monkeypatch.setenv("VOXGATE_JUDGE_PROVIDER", "bogus")

    with pytest.raises(ValueError):
        get_default_judge_client()
