from __future__ import annotations

import os
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, Optional, Sequence, Tuple

from cpdp.config import ExperimentConfig, load_config
from cpdp.methods.registry import METHOD_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.yaml"
ALL_TARGETS = "全部25个项目"
ALL_SEEDS = "全部配置的预测种子"
METHOD_LABELS = {
    "dssdpp": "DSSDPP",
    "master": "MASTER",
    "fedcpdp": "FedCPDP",
    "fd_hdp_u": "FD-HDP-U",
    "mbl_cpdp_sv": "MBL-CPDP-SV",
    "ipgdo_kan_jepa": "IPGDO + KAN-JEPA",
    "ipgdo_transformer_jepa": "IPGDO + Transformer-JEPA",
}


def default_seed_choice(profile: str, seeds: Sequence[int]) -> str:
    if not seeds:
        raise ValueError("At least one configured seed is required")
    if profile == "full":
        return ALL_SEEDS
    if profile == "smoke":
        return str(seeds[0])
    raise ValueError(f"Unknown profile: {profile}")


@dataclass(frozen=True)
class LaunchOptions:
    methods: Tuple[str, ...]
    profile: str
    target: Optional[str]
    seeds: Optional[Tuple[int, ...]]
    max_runs: Optional[int]
    workers: int
    force: bool


def seed_count_choice(count: int) -> str:
    if count < 1:
        raise ValueError("Seed count must be positive")
    return f"前{count}个种子"


def available_seed_choices(seeds: Sequence[int]) -> Tuple[str, ...]:
    if not seeds:
        raise ValueError("At least one configured seed is required")
    prefix_choices = tuple(
        seed_count_choice(count) for count in (5, 10, 20) if count < len(seeds)
    )
    return (ALL_SEEDS, *prefix_choices, *(str(seed) for seed in seeds))


def resolve_seed_choice(
    choice: str, seeds: Sequence[int]
) -> Optional[Tuple[int, ...]]:
    configured = tuple(int(seed) for seed in seeds)
    if choice == ALL_SEEDS:
        return None
    if choice.startswith("前") and choice.endswith("个种子"):
        count_text = choice[1:-3]
        if not count_text.isdigit():
            raise ValueError(f"无法识别种子数量：{choice}")
        count = int(count_text)
        if count < 1 or count > len(configured):
            raise ValueError(f"种子数量超出配置范围：{count}")
        return configured[:count]
    try:
        selected = int(choice)
    except ValueError as exc:
        raise ValueError(f"无法识别随机种子：{choice}") from exc
    if selected not in configured:
        raise ValueError(f"随机种子不在正式配置中：{selected}")
    return (selected,)


def build_run_command(
    python_executable: str, project_root: Path, options: LaunchOptions
) -> Sequence[str]:
    if not options.methods:
        raise ValueError("At least one method must be selected")
    if options.profile not in {"smoke", "full"}:
        raise ValueError(f"Unknown profile: {options.profile}")
    if options.workers < 1:
        raise ValueError("Parallel workers must be at least 1")
    method_value = (
        "all"
        if set(options.methods) == set(METHOD_NAMES)
        else ",".join(options.methods)
    )
    command = [
        python_executable,
        str(project_root / "run_experiment.py"),
        "run",
        "--method",
        method_value,
        "--profile",
        options.profile,
    ]
    if options.target is not None:
        command.extend(["--target", options.target])
    if options.seeds is not None:
        for seed in options.seeds:
            command.extend(["--seed", str(seed)])
    if options.max_runs is not None:
        command.extend(["--max-runs", str(options.max_runs)])
    command.extend(["--workers", str(options.workers)])
    if options.force:
        command.append("--force")
    return command


class ExperimentLauncher:
    def __init__(self, root: tk.Tk, config: ExperimentConfig) -> None:
        self.root = root
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None
        self.process_lock = threading.Lock()
        self.stop_requested = False
        self.is_running = False

        self.method_vars: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=name == "dssdpp") for name in METHOD_NAMES
        }
        self.profile_var = tk.StringVar(value="smoke")
        self.target_var = tk.StringVar(value="ReLink/Apache")
        self.seed_var = tk.StringVar(
            value=default_seed_choice(self.profile_var.get(), config.seeds)
        )
        self.max_runs_var = tk.StringVar(value="")
        recommended_workers = min(4, max(1, (os.cpu_count() or 1) // 4))
        self.workers_var = tk.StringVar(value=str(recommended_workers))
        self.force_var = tk.BooleanVar(value=False)
        self.auto_summary_var = tk.BooleanVar(value=True)
        self.task_text = tk.StringVar()
        self.status_text = tk.StringVar(value="就绪")

        self.target_choices = [ALL_TARGETS] + [
            f"{family}/{project}" for family, project in config.targets()
        ]
        self.configured_target_count = len(self.target_choices) - 1
        self.seed_choices = available_seed_choices(config.seeds)
        self._build_window()
        self._update_task_count()

    def _build_window(self) -> None:
        self.root.title("CPDP 实验启动器")
        self.root.geometry("980x720")
        self.root.minsize(860, 620)
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(3, weight=1)

        method_frame = ttk.LabelFrame(container, text="实验方法", padding=10)
        method_frame.grid(row=0, column=0, sticky="ew")
        method_columns = 2
        for index, name in enumerate(METHOD_NAMES):
            row, column = divmod(index, method_columns)
            checkbox = ttk.Checkbutton(
                method_frame,
                text=METHOD_LABELS[name],
                variable=self.method_vars[name],
                command=self._update_task_count,
            )
            checkbox.grid(row=row, column=column, padx=(0, 24), pady=2, sticky="w")
        button_row = (len(METHOD_NAMES) + method_columns - 1) // method_columns
        ttk.Button(method_frame, text="全选", command=self._select_all_methods).grid(
            row=button_row, column=0, padx=(0, 4), pady=(6, 0), sticky="w"
        )
        ttk.Button(method_frame, text="清空", command=self._clear_methods).grid(
            row=button_row, column=1, pady=(6, 0), sticky="w"
        )

        settings = ttk.LabelFrame(container, text="实验设置", padding=10)
        settings.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        for column in (1, 3, 5, 7):
            settings.columnconfigure(column, weight=1)

        ttk.Label(settings, text="模式").grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            settings,
            text="冒烟测试",
            value="smoke",
            variable=self.profile_var,
            command=self._on_profile_changed,
        ).grid(row=0, column=1, sticky="w")
        ttk.Radiobutton(
            settings,
            text="正式实验",
            value="full",
            variable=self.profile_var,
            command=self._on_profile_changed,
        ).grid(row=0, column=2, sticky="w", padx=(8, 18))

        ttk.Label(settings, text="目标项目").grid(row=0, column=3, sticky="e", padx=(0, 6))
        target_box = ttk.Combobox(
            settings,
            textvariable=self.target_var,
            values=self.target_choices,
            state="readonly",
            width=22,
        )
        target_box.grid(row=0, column=4, sticky="ew")
        target_box.bind("<<ComboboxSelected>>", lambda _: self._update_task_count())

        ttk.Label(settings, text="随机种子").grid(row=0, column=5, sticky="e", padx=(18, 6))
        seed_box = ttk.Combobox(
            settings,
            textvariable=self.seed_var,
            values=self.seed_choices,
            state="readonly",
            width=18,
        )
        seed_box.grid(row=0, column=6, sticky="ew")
        seed_box.bind("<<ComboboxSelected>>", lambda _: self._update_task_count())

        ttk.Label(settings, text="本次最多任务").grid(
            row=1, column=0, pady=(10, 0), sticky="w"
        )
        max_runs = ttk.Spinbox(
            settings,
            from_=1,
            to=max(1, len(METHOD_NAMES) * self.configured_target_count * len(self.config.seeds)),
            textvariable=self.max_runs_var,
            width=12,
            command=self._update_task_count,
        )
        max_runs.grid(row=1, column=1, pady=(10, 0), sticky="w")
        max_runs.bind("<KeyRelease>", lambda _: self._update_task_count())

        ttk.Checkbutton(
            settings,
            text="强制覆盖成功结果",
            variable=self.force_var,
        ).grid(row=1, column=2, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Checkbutton(
            settings,
            text="完成后自动汇总",
            variable=self.auto_summary_var,
        ).grid(row=1, column=4, columnspan=2, pady=(10, 0), sticky="w")
        ttk.Label(settings, textvariable=self.task_text).grid(
            row=1, column=6, columnspan=2, pady=(10, 0), sticky="e"
        )

        ttk.Label(settings, text="并行任务数").grid(
            row=2, column=0, pady=(10, 0), sticky="w"
        )
        worker_limit = max(1, min(8, os.cpu_count() or 1))
        workers = ttk.Spinbox(
            settings,
            from_=1,
            to=worker_limit,
            textvariable=self.workers_var,
            width=12,
            command=self._update_task_count,
        )
        workers.grid(row=2, column=1, pady=(10, 0), sticky="w")
        workers.bind("<KeyRelease>", lambda _: self._update_task_count())

        actions = ttk.Frame(container)
        actions.grid(row=2, column=0, pady=10, sticky="ew")
        self.start_button = ttk.Button(actions, text="启动实验", command=self._start_experiment)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions, text="停止", command=self._stop_process, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        self.validate_button = ttk.Button(
            actions, text="验证数据", command=self._validate_data
        )
        self.validate_button.pack(side="left", padx=(20, 0))
        self.summary_button = ttk.Button(
            actions, text="生成汇总", command=self._summarize
        )
        self.summary_button.pack(side="left", padx=(8, 0))
        ttk.Button(actions, text="打开结果目录", command=self._open_outputs).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="清空日志", command=self._clear_log).pack(side="right")

        log_frame = ttk.LabelFrame(container, text="运行日志", padding=6)
        log_frame.grid(row=3, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.log.grid(row=0, column=0, sticky="nsew")

        status = ttk.Label(container, textvariable=self.status_text, anchor="w")
        status.grid(row=4, column=0, pady=(8, 0), sticky="ew")

    def _selected_methods(self) -> Tuple[str, ...]:
        return tuple(name for name in METHOD_NAMES if self.method_vars[name].get())

    def _select_all_methods(self) -> None:
        for variable in self.method_vars.values():
            variable.set(True)
        self._update_task_count()

    def _clear_methods(self) -> None:
        for variable in self.method_vars.values():
            variable.set(False)
        self._update_task_count()

    def _parse_max_runs(self) -> Optional[int]:
        value = self.max_runs_var.get().strip()
        if not value:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError("本次最多任务必须大于 0")
        return parsed

    def _parse_workers(self) -> int:
        value = self.workers_var.get().strip()
        parsed = int(value)
        if parsed < 1:
            raise ValueError("并行任务数必须大于 0")
        if parsed > max(1, min(8, os.cpu_count() or 1)):
            raise ValueError("并行任务数超过当前启动器上限")
        return parsed

    def _update_task_count(self) -> None:
        method_count = len(self._selected_methods())
        target_count = (
            self.configured_target_count if self.target_var.get() == ALL_TARGETS else 1
        )
        selected_seeds = resolve_seed_choice(self.seed_var.get(), self.config.seeds)
        seed_count = len(self.config.seeds) if selected_seeds is None else len(selected_seeds)
        total = method_count * target_count * seed_count
        try:
            max_runs = self._parse_max_runs()
            workers = self._parse_workers()
        except ValueError:
            self.task_text.set("任务数：参数无效")
            return
        scheduled = min(total, max_runs) if max_runs is not None else total
        effective_workers = min(workers, max(scheduled, 1))
        self.task_text.set(
            f"本次任务上限：{scheduled} / 已选择：{total} / 并行：{effective_workers}"
        )

    def _on_profile_changed(self) -> None:
        profile = self.profile_var.get()
        self.seed_var.set(default_seed_choice(profile, self.config.seeds))
        if profile == "full":
            self.target_var.set(ALL_TARGETS)
        elif self.target_var.get() == ALL_TARGETS:
            self.target_var.set("ReLink/Apache")
        self._update_task_count()

    def _launch_options(self) -> LaunchOptions:
        target = None if self.target_var.get() == ALL_TARGETS else self.target_var.get()
        return LaunchOptions(
            methods=self._selected_methods(),
            profile=self.profile_var.get(),
            target=target,
            seeds=resolve_seed_choice(self.seed_var.get(), self.config.seeds),
            max_runs=self._parse_max_runs(),
            workers=self._parse_workers(),
            force=self.force_var.get(),
        )

    def _start_experiment(self) -> None:
        try:
            options = self._launch_options()
            command = build_run_command(sys.executable, PROJECT_ROOT, options)
        except (TypeError, ValueError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return
        if options.profile == "full":
            selected = self.task_text.get()
            confirmed = messagebox.askyesno(
                "启动正式实验",
                f"将使用 full 参数启动正式实验。\n{selected}\n\n确认继续？",
                parent=self.root,
            )
            if not confirmed:
                return
        self._start_command(
            command,
            title="实验",
            auto_summarize=self.auto_summary_var.get(),
        )

    def _validate_data(self) -> None:
        self._start_command(
            [sys.executable, str(PROJECT_ROOT / "run_experiment.py"), "validate-data"],
            title="数据验证",
        )

    def _summarize(self) -> None:
        self._start_command(
            [sys.executable, str(PROJECT_ROOT / "run_experiment.py"), "summarize"],
            title="结果汇总",
        )

    def _start_command(
        self,
        command: Sequence[str],
        *,
        title: str,
        auto_summarize: bool = False,
    ) -> None:
        if self.is_running:
            messagebox.showwarning("任务运行中", "请等待当前任务结束。", parent=self.root)
            return
        self.stop_requested = False
        self.is_running = True
        self._set_running(True)
        self._append_log(f"\n> {subprocess.list2cmdline(list(command))}\n")
        self.status_text.set(f"{title}运行中")
        worker = threading.Thread(
            target=self._run_worker,
            args=(tuple(command), title, auto_summarize),
            daemon=True,
        )
        worker.start()

    def _run_worker(
        self, command: Sequence[str], title: str, auto_summarize: bool
    ) -> None:
        return_code = self._stream_process(command)
        if auto_summarize and not self.stop_requested:
            self.root.after(0, self._append_log, "\n> 自动生成结果汇总\n")
            summary_code = self._stream_process(
                [sys.executable, str(PROJECT_ROOT / "run_experiment.py"), "summarize"]
            )
            if return_code == 0:
                return_code = summary_code
        self.root.after(0, self._finish_command, title, return_code)

    def _stream_process(self, command: Sequence[str]) -> int:
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        environment["PYTHONUNBUFFERED"] = "1"
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                list(command),
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=environment,
                creationflags=creation_flags,
            )
            with self.process_lock:
                self.process = process
            if self.stop_requested:
                process.terminate()
            if process.stdout is not None:
                for line in process.stdout:
                    self.root.after(0, self._append_log, line)
            return process.wait()
        except Exception as exc:
            self.root.after(0, self._append_log, f"启动失败：{type(exc).__name__}: {exc}\n")
            return 1
        finally:
            with self.process_lock:
                self.process = None

    def _finish_command(self, title: str, return_code: int) -> None:
        self.is_running = False
        self._set_running(False)
        if self.stop_requested:
            self.status_text.set(f"{title}已停止")
        elif return_code == 0:
            self.status_text.set(f"{title}完成")
        else:
            self.status_text.set(f"{title}结束，返回码 {return_code}")

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.validate_button.configure(state="disabled" if running else "normal")
        self.summary_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _stop_process(self) -> None:
        if not messagebox.askyesno(
            "停止任务", "确认停止当前实验进程？", parent=self.root
        ):
            return
        self.stop_requested = True
        self.status_text.set("正在停止")
        threading.Thread(target=self._terminate_process_tree, daemon=True).start()

    def _terminate_process_tree(self) -> None:
        with self.process_lock:
            process = self.process
        if process is None or process.poll() is not None:
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                capture_output=True,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        else:
            process.terminate()

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")

    def _open_outputs(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(self.config.output_dir)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(self.config.output_dir)])

    def _close_window(self) -> None:
        if self.is_running:
            close = messagebox.askyesno(
                "退出启动器",
                "实验仍在运行。退出将停止当前实验，确认继续？",
                parent=self.root,
            )
            if not close:
                return
            self.stop_requested = True
            self._terminate_process_tree()
        self.root.destroy()


def main() -> int:
    config = load_config(DEFAULT_CONFIG)
    root = tk.Tk()
    ExperimentLauncher(root, config)
    root.mainloop()
    return 0
