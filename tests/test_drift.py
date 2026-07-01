from eval_system.calibration.drift import detect_drift


def test_identical_distributions_show_no_drift():
    golden = [0.9, 0.85, 0.95, 0.88, 0.92, 0.91, 0.87, 0.93]
    current = [0.9, 0.85, 0.95, 0.88, 0.92, 0.91, 0.87, 0.93]

    result = detect_drift("faithfulness", golden, current)

    assert result.drifted is False
    assert result.p_value > 0.05


def test_clearly_shifted_distribution_flags_drift():
    golden = [0.9, 0.88, 0.92, 0.91, 0.89, 0.93, 0.87, 0.9]
    current = [0.3, 0.28, 0.32, 0.31, 0.29, 0.33, 0.27, 0.3]

    result = detect_drift("faithfulness", golden, current)

    assert result.drifted is True
    assert result.p_value < 0.05


def test_insufficient_samples_never_flags_drift():
    result = detect_drift("faithfulness", [0.9], [0.3])

    assert result.drifted is False


def test_threshold_is_configurable():
    golden = [0.9, 0.85, 0.95, 0.88, 0.92, 0.91, 0.87, 0.93]
    current = [0.86, 0.83, 0.9, 0.84, 0.88, 0.89, 0.85, 0.9]

    lenient = detect_drift("faithfulness", golden, current, p_value_threshold=0.01)
    strict = detect_drift("faithfulness", golden, current, p_value_threshold=0.999)

    # Same data, different threshold -- an absurdly high bar flags drift that
    # a strict one wouldn't.
    assert lenient.drifted is False
    assert strict.drifted is True
