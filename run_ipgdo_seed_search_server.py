from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
RUNTIME_DIR = PROJECT_ROOT / "outputs" / "ipgdo_seed_search" / "runtime"
PID_PATH = RUNTIME_DIR / "seed_search.pid"
LOG_PATH = RUNTIME_DIR / "seed_search.log"
RUNNER = PROJECT_ROOT / "run_ipgdo_seed_search.py"
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
        return "日志文件尚未生成。"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        content = handle.readlines()
    return "".join(content[-lines:]).rstrip()


def start(extra_arguments: Sequence[str]) -> int:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    current_pid = _read_pid()
    if _is_running(current_pid):
        print(f"种子搜索已经运行，PID={current_pid}")
        print(f"查看状态：{PYTHON} {Path(__file__).name} status")
        return 0
    if not PYTHON.is_file():
        raise FileNotFoundError(f"服务器虚拟环境不存在：{PYTHON}")

    environment = os.environ.copy()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[variable] = "1"
    command = [str(PYTHON), "-u", str(RUNNER), *extra_arguments]
    with LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(
            f"\nSERVER_SEARCH_START time={time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"command={' '.join(command)}\n"
        )
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
            close_fds=True,
        )
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"种子搜索已在后台启动，PID={process.pid}")
    print(f"日志：{LOG_PATH}")
    print(f"查看状态：{PYTHON} {Path(__file__).name} status")
    return 0


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
        print("种子搜索当前未运行。")
        return 0
    assert pid is not None
    os.killpg(pid, signal.SIGTERM)
    print(f"已请求停止种子搜索，PID={pid}。已完成折的结果会保留。")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="服务器IPGDO种子搜索后台控制器")
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start", help="后台启动或断点续跑")
    start_parser.add_argument(
        "runner_arguments",
        nargs=argparse.REMAINDER,
        help="传递给run_ipgdo_seed_search.py的参数",
    )
    status_parser = subparsers.add_parser("status", help="查看状态和最近日志")
    status_parser.add_argument("--lines", type=int, default=40)
    subparsers.add_parser("stop", help="安全停止后台实验")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "start":
        arguments = list(args.runner_arguments)
        if arguments[:1] == ["--"]:
            arguments = arguments[1:]
        return start(arguments)
    if args.command == "status":
        return status(args.lines)
    if args.command == "stop":
        return stop()
    raise AssertionError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
