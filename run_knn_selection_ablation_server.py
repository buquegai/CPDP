from __future__ import annotations

import argparse
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "knn_selection_ablation.yaml"
RUNTIME_DIR = PROJECT_ROOT / "outputs" / "knn_selection_ablation" / "runtime"
PID_PATH = RUNTIME_DIR / "knn_selection_ablation.pid"
LOG_PATH = RUNTIME_DIR / "knn_selection_ablation.log"
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _read_pid() -> Optional[int]:
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return None


def _is_running(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _tail(path: Path, lines: int) -> str:
    if not path.is_file():
        return "日志尚未生成。"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:]).rstrip()


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    source_path = str(PROJECT_ROOT / "src")
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path if not existing else f"{source_path}{os.pathsep}{existing}"
    )
    return environment


def _runner_command(extra_arguments: Sequence[str]) -> list[str]:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"未找到服务器虚拟环境：{PYTHON}")
    return [
        str(PYTHON),
        "-u",
        "-m",
        "cpdp.oversampling_validation.knn_selection_ablation",
        "run",
        "--config",
        str(CONFIG_PATH),
        *extra_arguments,
    ]


def start(extra_arguments: Sequence[str]) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    current_pid = _read_pid()
    if _is_running(current_pid):
        print(f"KNN选样消融实验正在运行：PID={current_pid}")
        return 0
    command = _runner_command(extra_arguments)
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(
            f"\nKNN_ABLATION_SERVER_START time={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"command={' '.join(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=_environment(),
            start_new_session=True,
            close_fds=True,
        )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"KNN选样消融实验已启动：PID={process.pid}")
    print(f"日志：{LOG_PATH}")
    return 0


def foreground(extra_arguments: Sequence[str]) -> int:
    completed = subprocess.run(
        _runner_command(extra_arguments),
        cwd=PROJECT_ROOT,
        env=_environment(),
        check=False,
    )
    return int(completed.returncode)


def status(lines: int) -> int:
    pid = _read_pid()
    state = "运行中" if _is_running(pid) else "未运行"
    print(f"状态：{state}")
    print(f"PID：{pid if pid is not None else '-'}")
    print(f"日志：{LOG_PATH}")
    print("\n最近日志：")
    print(_tail(LOG_PATH, lines))
    return 0


def stop() -> int:
    pid = _read_pid()
    if not _is_running(pid):
        print("KNN选样消融实验当前未运行。")
        return 0
    assert pid is not None
    os.killpg(pid, signal.SIGTERM)
    print(f"已向PID={pid}发送停止信号；已完成任务可断点续跑。")
    return 0


def summarize() -> int:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"未找到服务器虚拟环境：{PYTHON}")
    command = [
        str(PYTHON),
        "-u",
        "-m",
        "cpdp.oversampling_validation.knn_selection_ablation",
        "summarize",
        "--config",
        str(CONFIG_PATH),
    ]
    return int(
        subprocess.run(
            command, cwd=PROJECT_ROOT, env=_environment(), check=False
        ).returncode
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="服务器KNN选样消融实验控制器")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("start", "foreground"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("runner_arguments", nargs=argparse.REMAINDER)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--lines", type=int, default=60)
    subparsers.add_parser("stop")
    subparsers.add_parser("summarize")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command in {"start", "foreground"}:
        arguments = list(args.runner_arguments)
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        return (
            start(arguments)
            if args.command == "start"
            else foreground(arguments)
        )
    if args.command == "status":
        return status(args.lines)
    if args.command == "stop":
        return stop()
    if args.command == "summarize":
        return summarize()
    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
