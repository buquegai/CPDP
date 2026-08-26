from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)

# 每个搜索任务只使用一个BLAS/OpenMP线程，避免服务器容器因线程膨胀而终止进程。
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"


def main() -> int:
    if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
        return subprocess.call(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from cpdp.oversampling_seed_search.search import main as search_main

    return int(search_main(sys.argv[1:]))


if __name__ == "__main__":
    raise SystemExit(main())
