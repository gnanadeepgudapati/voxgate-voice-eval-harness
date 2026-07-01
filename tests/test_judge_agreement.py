from eval_system.calibration.judge_agreement import compute_judge_agreement


def test_perfect_agreement_is_trusted():
    judge = ["pass", "fail", "pass", "fail", "pass"]
    human = ["pass", "fail", "pass", "fail", "pass"]

    result = compute_judge_agreement("faithfulness", judge, human)

    assert result.kappa == 1.0
    assert result.trusted is True
    assert result.n == 5


def test_chance_level_agreement_is_not_trusted():
    # judge and human disagree in a pattern with no better-than-chance agreement
    judge = ["pass", "pass", "fail", "fail", "pass", "fail"]
    human = ["fail", "pass", "pass", "fail", "fail", "pass"]

    result = compute_judge_agreement("faithfulness", judge, human)

    assert result.trusted is False


def test_degenerate_single_class_data_is_never_trusted():
    # No variability in labels -- kappa is mathematically undefined (0/0), so
    # this must never be silently treated as "trusted".
    judge = ["pass", "pass", "pass"]
    human = ["pass", "pass", "pass"]

    result = compute_judge_agreement("faithfulness", judge, human)

    assert result.trusted is False
    assert result.kappa is None


def test_mismatched_lengths_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_judge_agreement("faithfulness", ["pass"], ["pass", "fail"])


def test_trust_threshold_is_configurable():
    judge = ["pass", "fail", "pass", "fail", "pass"]
    human = ["pass", "fail", "pass", "fail", "pass"]

    result = compute_judge_agreement("faithfulness", judge, human, kappa_threshold=1.5)

    assert result.trusted is False  # kappa of 1.0 doesn't clear an impossible 1.5 bar
