from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"


def _project_python() -> Path:
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def _running_in_project_environment(interpreter: Path) -> bool:
    return Path(sys.executable).resolve() == interpreter.resolve()


def main() -> int:
    interpreter = _project_python()
    if interpreter.is_file() and not _running_in_project_environment(interpreter):
        return subprocess.call(
            [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
    if not interpreter.is_file():
        print(f"未找到项目虚拟环境解释器：{interpreter}", file=sys.stderr)
        print("请先按照 README.md 安装 .venv。", file=sys.stderr)
        return 1

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    launcher = importlib.import_module("cpdp.launcher")
    if "--check" in sys.argv[1:]:
        print(f"LAUNCHER_OK interpreter={sys.executable}")
        return 0
    return int(launcher.main())


if __name__ == "__main__":
    raise SystemExit(main())
