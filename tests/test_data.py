from pathlib import Path

from cpdp.config import load_config
from cpdp.runner import validate_datasets


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_configured_25_projects_and_effort_columns_are_valid() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    projects, records = validate_datasets(config)

    assert len(projects) == 25
    assert len(records) == 25
    assert {record["family"] for record in records} == {
        "NASA",
        "AEEEM",
        "ReLink",
        "PROMISE",
    }
    assert all(record["rows"] > 1 for record in records)
    assert all(record["features"] > 0 for record in records)
    assert all(record["clean"] > 0 and record["defective"] > 0 for record in records)

    for family in config.data.families:
        schemas = {
            projects[f"{family.name}/{project}"].features.feature_names
            for project in family.projects
        }
        assert len(schemas) == 1
        assert all(
            projects[f"{family.name}/{project}"].features.effort_name
            == family.effort_column
            for project in family.projects
        )
