import numpy as np
import pytest

from cpdp.metrics import evaluate_predictions, popt_at_20


def test_six_metrics_on_known_predictions() -> None:
    labels = np.asarray([0, 0, 1, 1])
    scores = np.asarray([0.1, 0.4, 0.35, 0.8])
    effort = np.ones(4)

    metrics = evaluate_predictions(labels, scores, effort, threshold=0.5)

    assert metrics["auc"] == pytest.approx(0.75)
    assert metrics["auprc"] == pytest.approx(5.0 / 6.0)
    assert metrics["mcc"] == pytest.approx(1.0 / np.sqrt(3.0))
    assert metrics["f1"] == pytest.approx(2.0 / 3.0)
    assert metrics["gmean"] == pytest.approx(1.0 / np.sqrt(2.0))
    assert np.isfinite(metrics["popt20"])


def test_popt20_normalizes_perfect_and_worst_rankings() -> None:
    labels = np.asarray([0, 1, 0, 1])
    effort = np.ones(4)

    assert popt_at_20(labels, labels.astype(float), effort) == pytest.approx(1.0)
    assert popt_at_20(labels, 1.0 - labels, effort) == pytest.approx(0.0)
