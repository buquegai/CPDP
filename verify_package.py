from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cpdp.config import load_config
from cpdp.oversampling_validation.network_capacity_search import (
    NetworkCapacityParameterSet,
    load_network_capacity_search_config,
)


EXPECTED_FIXED = {
    "target_minority_ratio": 0.6,
    "candidate_multiplier": 3.5,
    "formula_l1_c": 0.1,
    "formula_max_terms": 10,
    "diffusion_epochs": 300,
    "diffusion_learning_rate": 0.001,
    "physics_warmup_fraction": 0.2,
    "formula_consistency_weight": 0.0,
    "formula_semantic_weight": 0.05,
    "defect_anchor_start_fraction": 0.5,
    "guidance_scale": 1.5,
    "physics_timestep_fraction": 0.4,
    "condition_dropout": 0.1,
    "manifold_lower_quantile": 0.025,
    "manifold_upper_quantile": 0.975,
    "manifold_neighbors": 5,
}


def _actual_fixed(search_config) -> dict[str, object]:
    fixed = search_config.fixed_overrides
    formula = fixed["formula"]
    diffusion = fixed["diffusion"]
    manifold = fixed["manifold"]
    return {
        "target_minority_ratio": fixed["target_minority_ratio"],
        "candidate_multiplier": fixed["candidate_multiplier"],
        "formula_l1_c": formula["l1_c"],
        "formula_max_terms": formula["max_terms"],
        "diffusion_epochs": diffusion["epochs"],
        "diffusion_learning_rate": diffusion["learning_rate"],
        "physics_warmup_fraction": diffusion["physics_warmup_fraction"],
        "formula_consistency_weight": diffusion["formula_consistency_weight"],
        "formula_semantic_weight": diffusion["formula_semantic_weight"],
        "defect_anchor_start_fraction": diffusion["defect_anchor_start_fraction"],
        "guidance_scale": diffusion["guidance_scale"],
        "physics_timestep_fraction": diffusion["physics_timestep_fraction"],
        "condition_dropout": diffusion["condition_dropout"],
        "manifold_lower_quantile": manifold["lower_quantile"],
        "manifold_upper_quantile": manifold["upper_quantile"],
        "manifold_neighbors": manifold["neighbors"],
    }


def main() -> int:
    experiment = load_config(PROJECT_ROOT / "configs" / "experiment.yaml")
    search = load_network_capacity_search_config(
        PROJECT_ROOT / "configs" / "ipgdo_network_capacity_search.yaml"
    )
    if experiment.data.root != (PROJECT_ROOT / "data").resolve():
        raise RuntimeError(f"数据目录不是包内路径: {experiment.data.root}")

    expected_files = {
        experiment.data.root / family.name / f"{project}{experiment.data.file_suffix}"
        for family in experiment.data.families
        for project in family.projects
    }
    missing = sorted(path for path in expected_files if not path.is_file())
    if missing:
        raise FileNotFoundError(f"缺少数据文件: {missing[0]}")
    if len(expected_files) != 25:
        raise RuntimeError(f"预期25个项目，实际配置为{len(expected_files)}个")
    if search.initial != NetworkCapacityParameterSet(256, 4, 200):
        raise RuntimeError(f"网络容量搜索基线异常: {search.initial}")
    if _actual_fixed(search) != EXPECTED_FIXED:
        raise RuntimeError("固定冠军参数与打包清单不一致")

    try:
        import torch

        device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        torch_text = f"torch={torch.__version__}, device={device}"
    except ImportError:
        torch_text = "torch=未安装（先运行setup_linux.sh）"

    print(
        "PACKAGE_OK "
        f"projects={len(expected_files)} "
        f"search_seeds={len(search.all_seeds)} "
        f"workers={search.execution.workers} "
        f"baseline={search.initial.trial_id} "
        f"{torch_text}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
