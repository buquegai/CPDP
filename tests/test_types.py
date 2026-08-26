from dataclasses import fields
import numpy as np

from cpdp.types import LabeledProject, ProjectFeatures, UnlabeledProject


def test_unlabeled_target_has_no_label_field() -> None:
    assert [field.name for field in fields(UnlabeledProject)] == ["features"]

    features = ProjectFeatures(
        family="demo",
        name="target",
        values=np.zeros((2, 1)),
        feature_names=("metric",),
        effort=np.ones(2),
        effort_name="metric",
        row_ids=np.asarray([3, 7]),
    )
    target = LabeledProject(features=features, labels=np.asarray([0, 1])).without_labels()

    assert not hasattr(features, "source_path")
    assert not hasattr(target, "labels")
    assert not hasattr(target.features, "source_path")
    assert target.features.row_ids.tolist() == [3, 7]
