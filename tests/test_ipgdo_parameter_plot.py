from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd

from cpdp.oversampling_validation.parameter_plot import (
    PARAMETER_SPECS,
    create_parameter_comparison_figure,
)


def _write_stage(output_dir: Path, key: str, filename: str) -> None:
    values = {
        "target_minority_ratio": (0.15, 0.25, 0.4, 0.6, 0.8),
        "physics_timestep_fraction": (0.25, 0.4, 0.5, 0.65, 0.8),
        "condition_dropout": (0.05, 0.1, 0.15, 0.2, 0.3),
    }[key]
    frame = pd.DataFrame(
        {
            key: values,
            "mean_rank": (3.4, 2.8, 2.5, 2.9, 3.2),
            "selection_position": (5, 2, 1, 3, 4),
        }
    )
    frame.to_csv(output_dir / filename, index=False)


def test_parameter_comparison_figure_is_created_and_nonblank(tmp_path: Path) -> None:
    for spec in PARAMETER_SPECS:
        _write_stage(tmp_path, spec.key, spec.stage_file)

    png_path, pdf_path = create_parameter_comparison_figure(
        tmp_path,
        {
            "target_minority_ratio": 0.4,
            "physics_timestep_fraction": 0.5,
            "condition_dropout": 0.15,
        },
        dpi=120,
    )

    assert png_path.is_file() and png_path.stat().st_size > 20_000
    assert pdf_path.is_file() and pdf_path.stat().st_size > 5_000
    image = mpimg.imread(png_path)
    assert image.shape[0] > 400 and image.shape[1] > 1_000
    assert float(np.std(image[..., :3])) > 0.02
