from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
BASE_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"


# =============================================================================
# 服务器实验参数区：通常只需要修改本区域，然后执行 python run_server_experiment.py
# =============================================================================

# 要运行的方法。可填写一个或多个；常用正式方法为 ipgdo_kan_jepa。
METHODS = ["ipgdo_kan_jepa"]

# full 为正式实验，smoke 仅用于快速检查代码能否运行。
PROFILE = "full"

# False 表示 smoke 始终使用基础配置中的轻量参数，避免误套用下方正式训练预算。
APPLY_PARAMETER_OVERRIDES_TO_SMOKE = False

# 从正式种子列表的第几个种子开始，按自然顺序编号，1 表示从第一个开始。
SEED_START_NUMBER = 1

# 本次连续运行多少个预测模型种子。例如 5、10；设为 None 表示运行到列表末尾。
# 这里仅控制预测模型种子。IPGDO按目标项目读取最低损失种子和增强数据缓存。
SEED_COUNT: int | None = 10

# 自定义种子。非空时优先使用这里的值，并忽略上面的起始位置和数量。
CUSTOM_SEEDS: list[int] = []

# 目标项目。空列表表示全部 25 个项目；也可填写 ["NASA/CM1", "ReLink/Apache"]。
TARGETS: list[str] = []

# 并行任务数。IPGDO 使用单张 GPU 时框架会自动限制为 1；CPU 方法可适当增大。
WORKERS = 1

# 限制本次最多执行多少个尚未完成的任务；None 表示不额外限制。
MAX_RUNS: int | None = None

# False 会自动跳过已经成功的同配置任务，实现断点续跑；True 会强制覆盖重跑。
FORCE = False

# 实验完成后是否自动生成 run_results、target_means 等汇总 CSV。
AUTO_SUMMARIZE = True

# 实验完成后是否自动汇总每个训练批次的预测损失并生成 PNG 曲线图。
AUTO_PLOT_PREDICTOR_LOSSES = True

PREDICTOR_BATCH_LOSS_METHODS = {
    "ipgdo_kan_jepa",
    "ipgdo_transformer_jepa",
}

# 预测模块的数据流模式：
# - "split"：JEPA 只学习真实源样本与无标签目标样本，分类器使用真实+生成源样本。
# - "shared_augmented_source"：数据不分流，真实+生成源样本同时用于 JEPA 和分类器训练。
PREDICTOR_DATA_FLOW_MODE = "shared_augmented_source"

# 结果分组名称，只能使用英文、数字、点、下划线和短横线。
RUN_GROUP = "formal"

# 代码或实验定义发生变化但参数未变化时，请手动递增此标签，避免复用旧结果。
EXPERIMENT_TAG = "v16-chebyshev-degree4-gmean-dynamic-lr-ratio0p7-loss-selected-seeds-ep160-300"


# IPGDO 的常用可调参数。键名采用“模块.子模块.参数”格式。
# 这里的值会覆盖 configs/experiment.yaml，但不会修改原始配置文件。
IPGDO_PARAMETER_OVERRIDES: dict[str, Any] = {
    # IPGDO 将每个源项目的缺陷/正常样本比例补齐到 1:1。
    "oversampling.target_minority_ratio": 0.7,
    # 自适应语义角色与稀疏公式挖掘参数。
    "oversampling.role.max_components": 4,
    "oversampling.role.l1_c": 0.2,
    "oversampling.role.minimum_component_weight": 0.01,
    "oversampling.formula.l1_c": 0.1,
    "oversampling.formula.max_terms": 10,
    "oversampling.formula.minimum_reliability": 0.0,
    "oversampling.formula.use_reliability_weighting": False,
    "oversampling.formula.project_blend_strength": 50.0,
    "oversampling.formula.minimum_class_samples": 4,
    # 项目条件表格 DDPM 及 PINN 物理损失参数。
    "oversampling.diffusion.hidden_dim": 256,
    "oversampling.diffusion.residual_blocks": 4,
    "oversampling.diffusion.timesteps": 200,
    # 必须与最低损失种子搜索保持一致，否则缓存完整性校验会拒绝读取。
    "oversampling.diffusion.epochs": 500,
    "oversampling.diffusion.learning_rate": 0.001,
    "oversampling.diffusion.physics_warmup_fraction": 0.2,
    "oversampling.diffusion.formula_consistency_weight": 0.0,
    "oversampling.diffusion.formula_semantic_weight": 0.05,
    # 数据不分流：真实源样本与 IPGDO 生成样本共同参与 JEPA 和分类器训练。
    "predictor.data_flow_mode": PREDICTOR_DATA_FLOW_MODE,
    # KAN-JEPA 预测模块训练预算
    "predictor.optimization.pretrain_epochs": 160,
    "predictor.optimization.joint_epochs": 300,
    # 动态学习率：2e-5 预热至 1.5e-4，再余弦衰减至 1e-5。
    "predictor.optimization.learning_rate": 0.0001,
    "predictor.optimization.learning_rate_schedule": "warmup_cosine",
    "predictor.optimization.learning_rate_warmup_start": 0.00002,
    "predictor.optimization.learning_rate_peak": 0.00015,
    "predictor.optimization.learning_rate_min": 0.00001,
    "predictor.optimization.learning_rate_warmup_fraction": 0.1,
    "predictor.optimization.learning_rate_decay_epochs": 50,
    # 已完成25x10复验的最佳阈值方案：源项目验证集G-mean最大化。
    "predictor.threshold_policy": "gmean",
    # G-mean最佳方案不进行概率校准。
    "predictor.probability_calibration": "none",
}


# 25个目标项目的最低损失种子及增强数据缓存映射。
# 完整实验会按目标项目名称自动读取对应条目，不再随预测种子重复训练IPGDO。
IPGDO_OVERSAMPLING_REUSE: dict[str, Any] = {
    "enabled": True,
    "seed_map": "artifacts/ipgdo_ratio0p7/best_oversampling_seeds.yaml",
}


# 每个方法可有独立参数。运行其他方法时，可按相同格式增加对应条目。
METHOD_PARAMETER_OVERRIDES: dict[str, dict[str, Any]] = {
    "ipgdo_kan_jepa": IPGDO_PARAMETER_OVERRIDES,
    # "master": {"lambda": 0.5, "smote_k_neighbors": 5},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="服务器 CPDP 实验快捷启动脚本")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只生成并检查运行配置，不启动正式实验",
    )
    return parser.parse_args()


def _project_venv_python() -> Path | None:
    candidates = (
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
    )
    return next((path for path in candidates if path.is_file()), None)


def _ensure_project_environment() -> int | None:
    venv_python = _project_venv_python()
    if venv_python is None:
        raise FileNotFoundError(
            "未找到项目虚拟环境。请先在项目根目录创建 .venv 并安装依赖。"
        )
    if Path(sys.prefix).resolve() == venv_python.parent.parent.resolve():
        return None
    completed = subprocess.run(
        [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"配置文件顶层必须是映射：{path}")
    # YAML 锚点可能让两个方法共享同一个字典对象；JSON 往返可解除这种引用共享。
    return json.loads(json.dumps(loaded, ensure_ascii=False))


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(
        dict(payload),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    os.replace(temporary, path)


def _set_existing_dotted_value(
    root: MutableMapping[str, Any], dotted_key: str, value: Any
) -> None:
    parts = dotted_key.split(".")
    if not all(parts):
        raise ValueError(f"参数路径格式错误：{dotted_key!r}")
    current: MutableMapping[str, Any] = root
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            raise KeyError(f"参数路径不存在：{dotted_key!r}（缺少 {part!r}）")
        current = child
    leaf = parts[-1]
    if leaf not in current:
        raise KeyError(f"参数路径不存在：{dotted_key!r}")
    current[leaf] = value


def _configured_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    experiment = config.get("experiment")
    if not isinstance(experiment, Mapping):
        raise KeyError("配置中缺少 experiment")
    raw_seeds = experiment.get("seeds")
    if not isinstance(raw_seeds, Sequence) or isinstance(raw_seeds, (str, bytes)):
        raise ValueError("experiment.seeds 必须是种子列表")
    seeds = tuple(int(seed) for seed in raw_seeds)
    if not seeds:
        raise ValueError("正式配置中至少需要一个随机种子")
    return seeds


def _selected_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    if CUSTOM_SEEDS:
        seeds = tuple(dict.fromkeys(int(seed) for seed in CUSTOM_SEEDS))
        if not seeds:
            raise ValueError("CUSTOM_SEEDS 不能只包含重复或无效值")
        return seeds

    configured = _configured_seeds(config)
    if SEED_START_NUMBER < 1 or SEED_START_NUMBER > len(configured):
        raise ValueError(
            f"SEED_START_NUMBER 必须位于 1 到 {len(configured)} 之间"
        )
    start = SEED_START_NUMBER - 1
    if SEED_COUNT is None:
        return configured[start:]
    if SEED_COUNT < 1:
        raise ValueError("SEED_COUNT 必须大于 0，或设为 None")
    selected = configured[start : start + SEED_COUNT]
    if len(selected) != SEED_COUNT:
        raise ValueError(
            f"从第 {SEED_START_NUMBER} 个种子开始只剩 {len(selected)} 个，"
            f"无法选择 {SEED_COUNT} 个"
        )
    return selected


def _validate_settings(config: Mapping[str, Any]) -> None:
    if not METHODS:
        raise ValueError("METHODS 至少需要包含一个方法")
    if len(set(METHODS)) != len(METHODS):
        raise ValueError("METHODS 中不能包含重复方法")
    if PROFILE not in {"full", "smoke"}:
        raise ValueError("PROFILE 只能是 'full' 或 'smoke'")
    if WORKERS < 1:
        raise ValueError("WORKERS 必须大于 0")
    if MAX_RUNS is not None and MAX_RUNS < 1:
        raise ValueError("MAX_RUNS 必须大于 0，或设为 None")
    if PREDICTOR_DATA_FLOW_MODE not in {"split", "shared_augmented_source"}:
        raise ValueError(
            "PREDICTOR_DATA_FLOW_MODE 只能是 'split' 或 "
            "'shared_augmented_source'"
        )
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", RUN_GROUP):
        raise ValueError("RUN_GROUP 只能包含英文、数字、点、下划线和短横线")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", EXPERIMENT_TAG):
        raise ValueError("EXPERIMENT_TAG 只能包含英文、数字、点、下划线和短横线")

    profiles = config.get("profiles")
    if not isinstance(profiles, Mapping) or not isinstance(profiles.get(PROFILE), Mapping):
        raise KeyError(f"配置中不存在 profile：{PROFILE}")
    profile_config = profiles[PROFILE]
    missing = [method for method in METHODS if method not in profile_config]
    if missing:
        raise KeyError(f"{PROFILE} 配置中不存在方法：{', '.join(missing)}")


def _apply_parameter_overrides(config: MutableMapping[str, Any]) -> None:
    if PROFILE == "smoke" and not APPLY_PARAMETER_OVERRIDES_TO_SMOKE:
        return
    profiles = config["profiles"]
    profile_config = profiles[PROFILE]
    for method, overrides in METHOD_PARAMETER_OVERRIDES.items():
        if method not in profile_config:
            raise KeyError(f"参数覆盖指定了不存在的方法：{PROFILE}/{method}")
        method_config = profile_config[method]
        for dotted_key, value in overrides.items():
            _set_existing_dotted_value(method_config, dotted_key, value)
    for method in METHODS:
        if method in {"ipgdo_kan_jepa", "ipgdo_transformer_jepa"}:
            profile_config[method]["oversampling_reuse"] = dict(
                IPGDO_OVERSAMPLING_REUSE
            )


def _validate_ipgdo_oversampling_reuse(config: Mapping[str, Any]) -> None:
    ipgdo_methods = {
        method
        for method in METHODS
        if method in {"ipgdo_kan_jepa", "ipgdo_transformer_jepa"}
    }
    if not ipgdo_methods or PROFILE != "full":
        return

    import yaml

    from cpdp.oversampling_seed_search.cache import mapping_signature

    raw_map_path = Path(str(IPGDO_OVERSAMPLING_REUSE["seed_map"]))
    map_path = (
        raw_map_path
        if raw_map_path.is_absolute()
        else (PROJECT_ROOT / raw_map_path).resolve()
    )
    if not map_path.is_file():
        raise FileNotFoundError(f"未找到IPGDO最低损失种子映射：{map_path}")
    with map_path.open("r", encoding="utf-8") as handle:
        seed_map = yaml.safe_load(handle)
    if not isinstance(seed_map, Mapping):
        raise ValueError(f"IPGDO最低损失种子映射格式错误：{map_path}")
    entries = seed_map.get("targets")
    if not isinstance(entries, Mapping):
        raise ValueError(f"IPGDO最低损失种子映射缺少targets：{map_path}")

    configured_targets = {
        f"{family}/{project}"
        for family, values in config["data"]["families"].items()
        for project in values["projects"]
    }
    requested_targets = set(TARGETS) if TARGETS else configured_targets
    missing_targets = sorted(requested_targets - set(entries))
    if missing_targets:
        raise KeyError(
            "IPGDO最低损失种子映射缺少目标项目："
            + ", ".join(missing_targets)
        )

    profile_config = config["profiles"][PROFILE]
    expected_signatures = {
        mapping_signature(profile_config[method]["oversampling"])
        for method in ipgdo_methods
    }
    if len(expected_signatures) != 1:
        raise ValueError("本轮IPGDO方法的过采样参数不一致，不能共享同一缓存")
    expected_signature = next(iter(expected_signatures))
    for target in sorted(requested_targets):
        entry = entries[target]
        if not isinstance(entry, Mapping):
            raise ValueError(f"IPGDO种子条目格式错误：{target}")
        cache_path = Path(str(entry.get("cache_file", "")))
        cache_path = (
            cache_path
            if cache_path.is_absolute()
            else (map_path.parent / cache_path).resolve()
        )
        manifest_path = cache_path.with_suffix(".json")
        if not cache_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"IPGDO缓存不完整：{target} -> {cache_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("oversampling_config_signature") != expected_signature:
            raise ValueError(
                f"IPGDO缓存参数与正式配置不一致：{target}。"
                "请重新搜索种子或恢复搜索时的过采样参数。"
            )


def _parameter_hash(config: Mapping[str, Any]) -> str:
    payload = {
        "experiment_tag": EXPERIMENT_TAG,
        "profile": PROFILE,
        "config": config,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _prepare_runtime_config() -> tuple[Path, Path, str, tuple[int, ...]]:
    if not BASE_CONFIG.is_file():
        raise FileNotFoundError(f"未找到基础配置：{BASE_CONFIG}")
    config = _load_yaml(BASE_CONFIG)
    _validate_settings(config)
    _apply_parameter_overrides(config)
    _validate_ipgdo_oversampling_reuse(config)
    seeds = _selected_seeds(config)

    experiment = config["experiment"]
    # 运行配置只声明本轮实际使用的预测种子，确保均值和完整性均按本轮种子数计算。
    experiment["seeds"] = list(seeds)
    experiment["repetitions"] = len(seeds)
    # output_dir 不参与参数哈希，避免结果目录本身改变哈希。
    experiment.pop("output_dir", None)
    parameter_hash = _parameter_hash(config)
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "server_runs"
        / RUN_GROUP
        / PROFILE
        / parameter_hash
    ).resolve()
    experiment["output_dir"] = str(output_dir)

    signature = f"server-{EXPERIMENT_TAG}-{parameter_hash}"
    profile_config = config["profiles"][PROFILE]
    for method in METHODS:
        profile_config[method]["result_signature"] = signature

    runtime_config = PROJECT_ROOT / "configs" / f"server_runtime_{parameter_hash}.yaml"
    _write_yaml(runtime_config, config)
    return runtime_config, output_dir, signature, seeds


def _run_command(runtime_config: Path, seeds: Sequence[int]) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "run_experiment.py"),
        "run",
        "--config",
        str(runtime_config),
        "--method",
        ",".join(METHODS),
        "--profile",
        PROFILE,
        "--workers",
        str(WORKERS),
    ]
    for target in TARGETS:
        command.extend(["--target", target])
    for seed in seeds:
        command.extend(["--seed", str(seed)])
    if MAX_RUNS is not None:
        command.extend(["--max-runs", str(MAX_RUNS)])
    if FORCE:
        command.append("--force")
    return command


def _summary_command(runtime_config: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "run_experiment.py"),
        "summarize",
        "--config",
        str(runtime_config),
    ]


def _prediction_loss_plot_command(output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "tools" / "summarize_prediction_batch_losses.py"),
        "--output-dir",
        str(output_dir),
    ]


def _should_plot_prediction_losses() -> bool:
    return AUTO_PLOT_PREDICTOR_LOSSES and any(
        method in PREDICTOR_BATCH_LOSS_METHODS for method in METHODS
    )


def _print_launch_summary(
    runtime_config: Path,
    output_dir: Path,
    signature: str,
    seeds: Sequence[int],
) -> None:
    targets_text = ", ".join(TARGETS) if TARGETS else "全部 25 个目标项目"
    print("=" * 78, flush=True)
    print("服务器 CPDP 实验", flush=True)
    print(f"方法             : {', '.join(METHODS)}", flush=True)
    print(f"模式             : {PROFILE}", flush=True)
    print(f"目标项目         : {targets_text}", flush=True)
    print(f"预测模型种子     : {len(seeds)} 个（{seeds[0]} ... {seeds[-1]}）", flush=True)
    print(f"并行任务数       : {WORKERS}", flush=True)
    flow_text = (
        "不分流（真实+生成源样本共同训练 JEPA 与分类器）"
        if PREDICTOR_DATA_FLOW_MODE == "shared_augmented_source"
        else "分流（JEPA 使用真实源样本与无标签目标样本）"
    )
    print(f"预测数据流       : {flow_text}", flush=True)
    runtime_payload = _load_yaml(runtime_config)
    optimization = runtime_payload["profiles"][PROFILE][METHODS[0]]["predictor"]["optimization"]
    if optimization.get("learning_rate_schedule") == "warmup_cosine":
        learning_rate_text = (
            "warmup_cosine（"
            f"{optimization['learning_rate_warmup_start']} -> "
            f"{optimization['learning_rate_peak']} -> "
            f"{optimization['learning_rate_min']}"
            ")"
        )
    else:
        learning_rate_text = str(optimization["learning_rate"])
    print(f"预测模型学习率   : {learning_rate_text}", flush=True)
    print(
        f"逐批次损失记录   : {'开启' if _should_plot_prediction_losses() else '关闭'}",
        flush=True,
    )
    if any(method in {"ipgdo_kan_jepa", "ipgdo_transformer_jepa"} for method in METHODS):
        print("IPGDO过采样种子  : 按目标项目读取最低损失种子", flush=True)
        print(
            f"IPGDO缓存映射    : {IPGDO_OVERSAMPLING_REUSE['seed_map']}",
            flush=True,
        )
    print(f"结果签名         : {signature}", flush=True)
    print(f"运行配置快照     : {runtime_config}", flush=True)
    print(f"结果目录         : {output_dir}", flush=True)
    print(f"强制重跑         : {'是' if FORCE else '否（自动续跑）'}", flush=True)
    print("=" * 78, flush=True)


def main() -> int:
    args = _parse_args()
    relaunched_exit_code = _ensure_project_environment()
    if relaunched_exit_code is not None:
        return relaunched_exit_code
    runtime_config, output_dir, signature, seeds = _prepare_runtime_config()
    _print_launch_summary(runtime_config, output_dir, signature, seeds)

    command = _run_command(runtime_config, seeds)
    print("启动命令：", " ".join(command), flush=True)
    if args.dry_run:
        print("配置检查完成；--dry-run 模式未启动实验。", flush=True)
        return 0

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False)
    if completed.returncode != 0:
        print(f"实验异常结束，退出码：{completed.returncode}", file=sys.stderr, flush=True)
        return completed.returncode

    if AUTO_SUMMARIZE:
        print("实验任务完成，正在生成汇总结果……", flush=True)
        summary = subprocess.run(
            _summary_command(runtime_config),
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if summary.returncode != 0:
            print(
                f"实验已完成，但汇总失败，退出码：{summary.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return summary.returncode

    if _should_plot_prediction_losses():
        print("正在汇总预测模型逐批次损失并生成曲线图……", flush=True)
        loss_summary = subprocess.run(
            _prediction_loss_plot_command(output_dir),
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
        )
        if loss_summary.returncode != 0:
            print(
                "实验已完成，但预测损失图生成失败，"
                f"退出码：{loss_summary.returncode}",
                file=sys.stderr,
                flush=True,
            )
            return loss_summary.returncode

    print(f"全部完成。结果保存在：{output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
