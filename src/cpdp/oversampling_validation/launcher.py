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

from cpdp.oversampling_validation.config import (
    OversamplingValidationConfig,
    load_validation_config,
)
from cpdp.oversampling_validation.runner import OversamplingValidationRunner


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "oversampling_validation.yaml"
ALL_FAMILIES = "全部数据集"
ALL_TARGETS = "全部目标项目"
ALL_SEEDS = "全部种子"
PROTOCOL_LABELS = {
    "source_cv": "源项目交叉验证",
    "cpdp": "跨项目 LOPO 验证",
}
VARIANT_LABELS = {
    "none": "不进行过采样",
    "random": "随机缺陷过采样",
    "smote": "SMOTE",
    "tabddpm": "TabDDPM",
    "full": "IPGDO（本文方法）",
}


@dataclass(frozen=True)
class LaunchOptions:
    profile: str
    protocols: Tuple[str, ...]
    variants: Tuple[str, ...]
    family: Optional[str]
    target: Optional[str]
    seeds: Tuple[int, ...]
    max_tasks: Optional[int]
    force: bool
    fail_fast: bool


def select_profile_seeds(
    configured_seeds: Sequence[int], selection: str, count: int
) -> Tuple[int, ...]:
    seeds = tuple(int(seed) for seed in configured_seeds)
    if not seeds:
        raise ValueError("当前配置档没有可用的随机种子")
    if selection != ALL_SEEDS:
        selected = int(selection)
        if selected not in seeds:
            raise ValueError("所选随机种子不属于当前配置档")
        return (selected,)
    if count < 1 or count > len(seeds):
        raise ValueError(f"运行种子数量必须在 1 到 {len(seeds)} 之间")
    return seeds[:count]


def build_run_command(
    python_executable: str,
    project_root: Path,
    options: LaunchOptions,
) -> Sequence[str]:
    if not options.protocols:
        raise ValueError("至少选择一种验证协议")
    if not options.variants:
        raise ValueError("至少选择一种过采样变体")
    if not options.seeds:
        raise ValueError("至少选择一个随机种子")
    command = [
        python_executable,
        str(project_root / "run_oversampling_validation.py"),
        "run",
        "--profile",
        options.profile,
    ]
    for protocol in options.protocols:
        command.extend(("--protocol", protocol))
    for variant in options.variants:
        command.extend(("--variant", variant))
    if options.family is not None:
        command.extend(("--family", options.family))
    if options.target is not None:
        command.extend(("--target", options.target))
    for seed in options.seeds:
        command.extend(("--seed", str(seed)))
    if options.max_tasks is not None:
        command.extend(("--max-tasks", str(options.max_tasks)))
    if options.force:
        command.append("--force")
    if options.fail_fast:
        command.append("--fail-fast")
    return command


class OversamplingValidationLauncher:
    def __init__(self, root: tk.Tk, config: OversamplingValidationConfig) -> None:
        self.root = root
        self.config = config
        self.process: Optional[subprocess.Popen[str]] = None
        self.process_lock = threading.Lock()
        self.stop_requested = False
        self.is_running = False

        default_profile = "smoke" if "smoke" in config.profiles else next(iter(config.profiles))
        profile = config.profile(default_profile)
        self.profile_var = tk.StringVar(value=default_profile)
        self.protocol_vars: Dict[str, tk.BooleanVar] = {
            "source_cv": tk.BooleanVar(value=profile.source_cv.enabled),
            "cpdp": tk.BooleanVar(
                value=profile.cpdp.enabled and not profile.source_cv.enabled
            ),
        }
        self.variant_vars: Dict[str, tk.BooleanVar] = {
            name: tk.BooleanVar(value=name in profile.variants)
            for name in config.variants
        }
        self.variant_buttons: Dict[str, ttk.Checkbutton] = {}
        self.family_var = tk.StringVar(value=self._default_family())
        self.target_var = tk.StringVar(value=ALL_TARGETS)
        self.seed_var = tk.StringVar(value=str(profile.seeds[0]))
        self.seed_count_var = tk.StringVar(value=str(len(profile.seeds)))
        self.max_tasks_var = tk.StringVar(value="")
        self.force_var = tk.BooleanVar(value=False)
        self.fail_fast_var = tk.BooleanVar(value=True)
        self.task_text = tk.StringVar()
        self.status_text = tk.StringVar(value="就绪")

        self.family_choices = [ALL_FAMILIES] + [
            family.name for family in config.experiment.data.families
        ]
        self.target_choices = [ALL_TARGETS] + [
            f"{family}/{project}" for family, project in config.experiment.targets()
        ]
        self._build_window()
        self._update_profile_controls()
        self._update_task_count()

    def _default_family(self) -> str:
        names = [family.name for family in self.config.experiment.data.families]
        return "NASA" if "NASA" in names else names[0]

    def _build_window(self) -> None:
        self.root.title("CPDP 过采样模块验证")
        self.root.geometry("1040x760")
        self.root.minsize(900, 650)
        self.root.protocol("WM_DELETE_WINDOW", self._close_window)

        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")

        container = ttk.Frame(self.root, padding=12)
        container.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(4, weight=1)

        variants = ttk.LabelFrame(container, text="过采样变体", padding=10)
        variants.grid(row=0, column=0, sticky="ew")
        for index, (name, variable) in enumerate(self.variant_vars.items()):
            button = ttk.Checkbutton(
                variants,
                text=VARIANT_LABELS.get(name, name),
                variable=variable,
                command=self._update_task_count,
            )
            button.grid(row=index // 3, column=index % 3, padx=(0, 24), pady=2, sticky="w")
            self.variant_buttons[name] = button
        button_row = (len(self.variant_vars) + 2) // 3
        ttk.Button(variants, text="全选", command=self._select_all_variants).grid(
            row=button_row, column=0, pady=(6, 0), sticky="w"
        )
        ttk.Button(variants, text="清空", command=self._clear_variants).grid(
            row=button_row, column=1, pady=(6, 0), sticky="w"
        )

        protocols = ttk.LabelFrame(container, text="验证协议", padding=10)
        protocols.grid(row=1, column=0, pady=(10, 0), sticky="ew")
        for index, (name, variable) in enumerate(self.protocol_vars.items()):
            ttk.Checkbutton(
                protocols,
                text=PROTOCOL_LABELS[name],
                variable=variable,
                command=self._update_task_count,
            ).grid(row=0, column=index, padx=(0, 28), sticky="w")

        settings = ttk.LabelFrame(container, text="实验设置", padding=10)
        settings.grid(row=2, column=0, pady=(10, 0), sticky="ew")
        for column in (1, 3, 5):
            settings.columnconfigure(column, weight=1)

        ttk.Label(settings, text="配置档").grid(row=0, column=0, sticky="w")
        profile_box = ttk.Combobox(
            settings,
            textvariable=self.profile_var,
            values=list(self.config.profiles),
            state="readonly",
            width=12,
        )
        profile_box.grid(row=0, column=1, sticky="ew")
        profile_box.bind("<<ComboboxSelected>>", self._on_profile_changed)

        ttk.Label(settings, text="数据集").grid(row=0, column=2, padx=(18, 6), sticky="e")
        family_box = ttk.Combobox(
            settings,
            textvariable=self.family_var,
            values=self.family_choices,
            state="readonly",
            width=18,
        )
        family_box.grid(row=0, column=3, sticky="ew")
        family_box.bind("<<ComboboxSelected>>", self._on_family_changed)

        ttk.Label(settings, text="目标项目").grid(row=0, column=4, padx=(18, 6), sticky="e")
        self.target_box = ttk.Combobox(
            settings,
            textvariable=self.target_var,
            values=self.target_choices,
            state="readonly",
            width=23,
        )
        self.target_box.grid(row=0, column=5, sticky="ew")
        self.target_box.bind("<<ComboboxSelected>>", self._on_target_changed)

        ttk.Label(settings, text="随机种子").grid(row=1, column=0, pady=(10, 0), sticky="w")
        self.seed_box = ttk.Combobox(
            settings,
            textvariable=self.seed_var,
            state="readonly",
            width=16,
        )
        self.seed_box.grid(row=1, column=1, pady=(10, 0), sticky="ew")
        self.seed_box.bind("<<ComboboxSelected>>", self._on_seed_changed)

        ttk.Label(settings, text="运行种子数量").grid(
            row=1, column=2, padx=(18, 6), pady=(10, 0), sticky="e"
        )
        self.seed_count_box = ttk.Spinbox(
            settings,
            from_=1,
            to=max(1, len(self.config.profile(self.profile_var.get()).seeds)),
            textvariable=self.seed_count_var,
            width=14,
            command=self._update_task_count,
        )
        self.seed_count_box.grid(row=1, column=3, pady=(10, 0), sticky="w")
        self.seed_count_box.bind("<KeyRelease>", lambda _: self._update_task_count())

        ttk.Label(settings, text="本次最多任务").grid(
            row=1, column=4, padx=(18, 6), pady=(10, 0), sticky="e"
        )
        max_tasks = ttk.Spinbox(
            settings,
            from_=1,
            to=100000,
            textvariable=self.max_tasks_var,
            width=14,
            command=self._update_task_count,
        )
        max_tasks.grid(row=1, column=5, pady=(10, 0), sticky="w")
        max_tasks.bind("<KeyRelease>", lambda _: self._update_task_count())

        toggles = ttk.Frame(settings)
        toggles.grid(row=2, column=0, columnspan=6, pady=(10, 0), sticky="w")
        ttk.Checkbutton(
            toggles,
            text="强制重跑已有结果",
            variable=self.force_var,
            command=self._update_task_count,
        ).pack(side="left")
        ttk.Checkbutton(
            toggles,
            text="失败时停止",
            variable=self.fail_fast_var,
        ).pack(side="left", padx=(14, 0))

        ttk.Label(settings, textvariable=self.task_text).grid(
            row=3, column=0, columnspan=6, pady=(10, 0), sticky="w"
        )

        actions = ttk.Frame(container)
        actions.grid(row=3, column=0, pady=10, sticky="ew")
        self.start_button = ttk.Button(actions, text="启动验证", command=self._start_validation)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(
            actions, text="停止", command=self._stop_process, state="disabled"
        )
        self.stop_button.pack(side="left", padx=(8, 0))
        self.summary_button = ttk.Button(actions, text="重新生成汇总", command=self._summarize)
        self.summary_button.pack(side="left", padx=(20, 0))
        ttk.Button(actions, text="打开结果目录", command=self._open_outputs).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(actions, text="清空日志", command=self._clear_log).pack(side="right")

        log_frame = ttk.LabelFrame(container, text="运行日志", padding=6)
        log_frame.grid(row=4, column=0, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            state="disabled",
            font=("Consolas", 10),
        )
        self.log.grid(row=0, column=0, sticky="nsew")
        ttk.Label(container, textvariable=self.status_text, anchor="w").grid(
            row=5, column=0, pady=(8, 0), sticky="ew"
        )

    def _selected_variants(self) -> Tuple[str, ...]:
        enabled = set(self.config.profile(self.profile_var.get()).variants)
        return tuple(
            name for name, variable in self.variant_vars.items()
            if name in enabled and variable.get()
        )

    def _selected_protocols(self) -> Tuple[str, ...]:
        profile = self.config.profile(self.profile_var.get())
        enabled = {
            "source_cv": profile.source_cv.enabled,
            "cpdp": profile.cpdp.enabled,
        }
        return tuple(
            name for name, variable in self.protocol_vars.items()
            if enabled[name] and variable.get()
        )

    def _select_all_variants(self) -> None:
        enabled = set(self.config.profile(self.profile_var.get()).variants)
        for name, variable in self.variant_vars.items():
            variable.set(name in enabled)
        self._update_task_count()

    def _clear_variants(self) -> None:
        for variable in self.variant_vars.values():
            variable.set(False)
        self._update_task_count()

    def _parse_max_tasks(self) -> Optional[int]:
        value = self.max_tasks_var.get().strip()
        if not value:
            return None
        parsed = int(value)
        if parsed < 1:
            raise ValueError("本次最多任务必须大于 0")
        return parsed

    def _selected_seeds(self) -> Tuple[int, ...]:
        profile = self.config.profile(self.profile_var.get())
        selection = self.seed_var.get()
        count = 1 if selection != ALL_SEEDS else int(self.seed_count_var.get())
        return select_profile_seeds(profile.seeds, selection, count)

    def _launch_options(self) -> LaunchOptions:
        target = None if self.target_var.get() == ALL_TARGETS else self.target_var.get()
        family = None if self.family_var.get() == ALL_FAMILIES else self.family_var.get()
        if target is not None:
            target_family = target.split("/", 1)[0]
            if family is not None and family != target_family:
                raise ValueError("目标项目必须属于当前选择的数据集")
        return LaunchOptions(
            profile=self.profile_var.get(),
            protocols=self._selected_protocols(),
            variants=self._selected_variants(),
            family=family,
            target=target,
            seeds=self._selected_seeds(),
            max_tasks=self._parse_max_tasks(),
            force=self.force_var.get(),
            fail_fast=self.fail_fast_var.get(),
        )

    def _estimate_task_count(self) -> int:
        profile = self.config.profile(self.profile_var.get())
        variant_count = len(self._selected_variants())
        seed_count = len(self._selected_seeds())
        selected_family = self.family_var.get()
        family_count = (
            len(self.config.experiment.data.families)
            if selected_family == ALL_FAMILIES else 1
        )
        target = self.target_var.get()
        if target != ALL_TARGETS:
            target_count = 1
        elif selected_family == ALL_FAMILIES:
            target_count = sum(
                len(family.projects) for family in self.config.experiment.data.families
            )
        else:
            target_count = len(self.config.experiment.family(selected_family).projects)
        total = 0
        protocols = self._selected_protocols()
        if "source_cv" in protocols:
            total += variant_count * family_count * seed_count * profile.source_cv.folds
        if "cpdp" in protocols:
            total += variant_count * target_count * seed_count
        return total

    def _update_task_count(self) -> None:
        try:
            selected = self._estimate_task_count()
            selected_seed_count = len(self._selected_seeds())
            configured_seed_count = len(
                self.config.profile(self.profile_var.get()).seeds
            )
            maximum = self._parse_max_tasks()
            scheduled = min(selected, maximum) if maximum is not None else selected
            suffix = "；已有成功结果默认跳过" if not self.force_var.get() else "；将强制重跑"
            self.task_text.set(
                f"种子数量：{selected_seed_count} / {configured_seed_count}；"
                f"本次任务上限：{scheduled} / 当前选择：{selected}{suffix}"
            )
        except (TypeError, ValueError):
            self.task_text.set("任务数量：参数无效")

    def _update_profile_controls(self) -> None:
        profile = self.config.profile(self.profile_var.get())
        seed_values = [ALL_SEEDS] + [str(seed) for seed in profile.seeds]
        self.seed_box.configure(values=seed_values)
        use_all_seeds = self.profile_var.get() == "full"
        self.seed_var.set(ALL_SEEDS if use_all_seeds else str(profile.seeds[0]))
        self.seed_count_var.set(str(len(profile.seeds) if use_all_seeds else 1))
        self.seed_count_box.configure(
            to=max(1, len(profile.seeds)),
            state="normal" if use_all_seeds else "disabled",
        )
        for name, variable in self.variant_vars.items():
            enabled = name in profile.variants
            self.variant_buttons[name].configure(state="normal" if enabled else "disabled")
            if not enabled:
                variable.set(False)
        if not profile.source_cv.enabled:
            self.protocol_vars["source_cv"].set(False)
        if not profile.cpdp.enabled:
            self.protocol_vars["cpdp"].set(False)

    def _on_profile_changed(self, _: object = None) -> None:
        self._update_profile_controls()
        self._update_task_count()

    def _on_seed_changed(self, _: object = None) -> None:
        profile = self.config.profile(self.profile_var.get())
        use_all_seeds = self.seed_var.get() == ALL_SEEDS
        self.seed_count_var.set(str(len(profile.seeds) if use_all_seeds else 1))
        self.seed_count_box.configure(state="normal" if use_all_seeds else "disabled")
        self._update_task_count()

    def _on_family_changed(self, _: object = None) -> None:
        family = self.family_var.get()
        if family == ALL_FAMILIES:
            values = self.target_choices
        else:
            values = [ALL_TARGETS] + [
                f"{family}/{project}"
                for project in self.config.experiment.family(family).projects
            ]
        self.target_box.configure(values=values)
        if self.target_var.get() not in values:
            self.target_var.set(ALL_TARGETS)
        self._update_task_count()

    def _on_target_changed(self, _: object = None) -> None:
        target = self.target_var.get()
        if target != ALL_TARGETS:
            self.family_var.set(target.split("/", 1)[0])
            self._on_family_changed()
            self.target_var.set(target)
        self._update_task_count()

    def _start_validation(self) -> None:
        try:
            options = self._launch_options()
            command = build_run_command(sys.executable, PROJECT_ROOT, options)
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.root)
            return
        if options.profile == "full":
            confirmed = messagebox.askyesno(
                "启动正式验证",
                f"将使用 full 参数启动正式验证。\n{self.task_text.get()}\n\n确认继续？",
                parent=self.root,
            )
            if not confirmed:
                return
        self._start_command(command, "过采样验证")

    def _summarize(self) -> None:
        self._start_command(
            [
                sys.executable,
                str(PROJECT_ROOT / "run_oversampling_validation.py"),
                "summarize",
                "--profile",
                self.profile_var.get(),
            ],
            "结果汇总",
        )

    def _start_command(self, command: Sequence[str], title: str) -> None:
        if self.is_running:
            messagebox.showwarning("任务运行中", "请等待当前任务结束", parent=self.root)
            return
        self.stop_requested = False
        self.is_running = True
        self._set_running(True)
        self._append_log(f"\n> {subprocess.list2cmdline(list(command))}\n")
        self.status_text.set(f"{title}运行中")
        threading.Thread(
            target=self._run_worker,
            args=(tuple(command), title),
            daemon=True,
        ).start()

    def _run_worker(self, command: Sequence[str], title: str) -> None:
        return_code = self._stream_process(command)
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
        self.summary_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def _stop_process(self) -> None:
        if not messagebox.askyesno(
            "停止任务", "确认停止当前验证进程？", parent=self.root
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
        profile = self.profile_var.get()
        try:
            fingerprint = OversamplingValidationRunner(self.config).fingerprint(profile)
            directory = self.config.output_dir / fingerprint / profile
        except (FileNotFoundError, KeyError, ValueError):
            directory = self.config.output_dir
        directory.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(directory)  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(directory)])

    def _close_window(self) -> None:
        if self.is_running:
            close = messagebox.askyesno(
                "退出启动器",
                "验证仍在运行。退出将停止当前任务，确认继续？",
                parent=self.root,
            )
            if not close:
                return
            self.stop_requested = True
            self._terminate_process_tree()
        self.root.destroy()


def main() -> int:
    config = load_validation_config(DEFAULT_CONFIG)
    root = tk.Tk()
    OversamplingValidationLauncher(root, config)
    root.mainloop()
    return 0
