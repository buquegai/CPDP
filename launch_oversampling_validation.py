from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
# 弹性调度只改变任务启动时机，不改变模型、数据、随机种子和训练参数。
ELASTIC_GPU_SCHEDULING = True
# 最大并发上限。实际并发数会根据 GPU 状态在 1 到该值之间自动变化。
DEFAULT_PARALLEL_WORKERS = 4
# GPU 利用率达到该值后暂停派发新任务，已启动任务继续运行。
GPU_LAUNCH_UTILIZATION_LIMIT = 75
# GPU 空闲显存低于该值（MiB）时暂停派发新任务。
GPU_MIN_FREE_MEMORY_MB = 2048
# GPU 温度达到该值（摄氏度）时暂停派发新任务。
GPU_MAX_TEMPERATURE_C = 80
# GPU 状态轮询间隔（秒）。
GPU_POLL_SECONDS = 2
# 每启动一个任务后的观察时间（秒），避免瞬间启动过多任务。
GPU_RAMP_UP_SECONDS = 8
# 可填写游戏主程序名，例如 ("game.exe",)。检测到后暂停派发新任务。
# 不建议填写长期驻留的游戏平台启动器，否则任务可能一直等待。
GPU_PAUSE_PROCESS_NAMES = ()


def _project_python() -> Path:
    if os.name == "nt":
        return PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return PROJECT_ROOT / ".venv" / "bin" / "python"


def main() -> int:
    elastic_settings = {
        "CPDP_OVERSAMPLING_VALIDATION_WORKERS": str(DEFAULT_PARALLEL_WORKERS),
        "CPDP_OVERSAMPLING_ELASTIC_GPU": "1" if ELASTIC_GPU_SCHEDULING else "0",
        "CPDP_GPU_LAUNCH_UTILIZATION_LIMIT": str(GPU_LAUNCH_UTILIZATION_LIMIT),
        "CPDP_GPU_MIN_FREE_MEMORY_MB": str(GPU_MIN_FREE_MEMORY_MB),
        "CPDP_GPU_MAX_TEMPERATURE_C": str(GPU_MAX_TEMPERATURE_C),
        "CPDP_GPU_POLL_SECONDS": str(GPU_POLL_SECONDS),
        "CPDP_GPU_RAMP_UP_SECONDS": str(GPU_RAMP_UP_SECONDS),
        "CPDP_GPU_PAUSE_PROCESSES": ";".join(GPU_PAUSE_PROCESS_NAMES),
    }
    for name, value in elastic_settings.items():
        os.environ[name] = value
    interpreter = _project_python()
    if not interpreter.is_file():
        print(f"Project virtual-environment interpreter not found: {interpreter}", file=sys.stderr)
        return 1
    if Path(sys.executable).resolve() != interpreter.resolve():
        return subprocess.call(
            [str(interpreter), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )

    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    launcher = importlib.import_module("cpdp.oversampling_validation.launcher")
    if "--check" in sys.argv[1:]:
        print(
            "OVERSAMPLING_VALIDATION_LAUNCHER_OK "
            f"interpreter={sys.executable} workers={DEFAULT_PARALLEL_WORKERS} "
            f"elastic_gpu={ELASTIC_GPU_SCHEDULING}"
        )
        return 0
    return int(launcher.main())


if __name__ == "__main__":
    raise SystemExit(main())
