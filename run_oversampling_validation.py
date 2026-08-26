from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
VENV_PYTHON = PROJECT_ROOT / ".venv" / (
    "Scripts/python.exe" if os.name == "nt" else "bin/python"
)
if VENV_PYTHON.is_file() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    raise SystemExit(
        subprocess.call(
            [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=PROJECT_ROOT,
        )
    )

SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oversampling_validation_parallel import (  # noqa: E402
    build_parallel_phases,
    configured_default_workers,
    configured_elastic_gpu,
    parse_parallel_control,
    parse_run_selection,
    run_parallel_validation,
)
from cpdp.oversampling_validation.cli import main  # noqa: E402


if __name__ == "__main__":
    try:
        control = parse_parallel_control(
            sys.argv[1:], configured_default_workers(fallback=4)
        )
        selection = parse_run_selection(control.argv, PROJECT_ROOT)
        elastic_gpu = configured_elastic_gpu(control.workers)
        if control.worker_process and selection is not None:
            from cpdp.oversampling_validation.store import ValidationStore

            def _defer_summary_until_parent(_store):
                return {}

            ValidationStore.summarize = _defer_summary_until_parent
        use_parallel = (
            not control.worker_process
            and (control.workers > 1 or elastic_gpu.enabled)
            and selection is not None
            and selection.max_tasks is None
            and bool(
                build_parallel_phases(
                    control.argv,
                    PROJECT_ROOT,
                    control.workers,
                    atomic=elastic_gpu.enabled,
                )
            )
        )
        if use_parallel:
            raise SystemExit(
                run_parallel_validation(
                    control.argv,
                    workers=control.workers,
                    project_root=PROJECT_ROOT,
                    python_executable=Path(sys.executable),
                    runner_script=Path(__file__).resolve(),
                    elastic_gpu=elastic_gpu,
                )
            )
        if selection is not None and selection.max_tasks is not None and control.workers > 1:
            print(
                "PARALLEL disabled because --max-tasks requires the original global "
                "serial ordering; use --workers 1 or remove --max-tasks.",
                flush=True,
            )
        raise SystemExit(main(control.argv))
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Parallel launcher error: {exc}", file=sys.stderr)
        raise SystemExit(2)
