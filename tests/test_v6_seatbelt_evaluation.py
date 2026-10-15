from training.evaluate_seatbelt_v6 import evaluate_rows, expected_calibration_error


def _row(true_class: str, probabilities: tuple[float, float, float]) -> dict[str, str]:
    return {
        "true_class": true_class,
        "prob_fastened": str(probabilities[0]),
        "prob_unfastened": str(probabilities[1]),
        "prob_uncertain": str(probabilities[2]),
    }


def test_policy_rejects_low_confidence_unfastened() -> None:
    rows = [
        _row("seatbelt_fastened", (0.40, 0.55, 0.05)),
        _row("seatbelt_unfastened", (0.10, 0.80, 0.10)),
        _row("uncertain_or_occluded", (0.10, 0.20, 0.70)),
    ]

    result = evaluate_rows(rows)

    assert result["abstention"]["threshold_forced_count"] == 1
    assert result["false_unfastened"]["count"] == 0
    assert result["post_policy"]["confusion_matrix"]["seatbelt_fastened"][
        "uncertain_or_occluded"
    ] == 1
    assert result["post_policy"]["accuracy"] == 0.666667


def test_expected_calibration_error_matches_simple_case() -> None:
    assert expected_calibration_error([0.8, 0.8], [True, False], bins=10) == 0.3
