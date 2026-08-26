from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)


def main() -> int:
    if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        return subprocess.call(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    # 这些参数只控制任务调度，不改变模型训练、随机种子或指标计算。
    os.environ.setdefault("CPDP_GPU_MAX_CONCURRENT_TASKS", "4")
    os.environ.setdefault("CPDP_GPU_TRANSIENT_RETRIES", "3")
    os.environ.setdefault("CPDP_GPU_RETRY_COOLDOWN_SECONDS", "30")
    from parameter_search_runtime_patch import install_parameter_search_runtime_patch

    resume_signature = install_parameter_search_runtime_patch(
        PROJECT_ROOT,
        state_path=(
            PROJECT_ROOT
            / "outputs"
            / "ipgdo_formula_optimization_search"
            / "search_state.json"
        ),
    )
    if resume_signature is not None:
        print(
            f"PARAMETER_SEARCH_RESUME source_signature={resume_signature}",
            flush=True,
        )
    from cpdp.oversampling_validation.formula_optimization_search import (
        main as search_main,
    )

    return int(search_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
