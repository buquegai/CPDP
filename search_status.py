from __future__ import annotations

import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ipgdo_network_capacity_search"


def main() -> int:
    state_path = OUTPUT_DIR / "search_state.json"
    plan_path = OUTPUT_DIR / "search_plan.yaml"
    if plan_path.is_file():
        plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
        print(
            "计划: "
            f"最多{plan['maximum_unique_configurations']}组参数，"
            f"最多{plan['maximum_total_tasks']}个任务"
        )
    else:
        print("计划: 尚未生成，请先运行 bash plan.sh")

    if not state_path.is_file():
        print("状态: 尚未开始正式搜索")
        return 0

    state = json.loads(state_path.read_text(encoding="utf-8"))
    print(f"状态: {state.get('status', 'unknown')}")
    print(f"开始时间: {state.get('started_at') or '-'}")
    print(f"更新时间: {state.get('updated_at') or '-'}")
    stages = state.get("stages", {})
    if stages:
        print("阶段:")
        for name, stage in stages.items():
            print(
                f"  {name}: {stage.get('status', 'unknown')}, "
                f"候选={stage.get('candidate_count', '-')}, "
                f"冠军={stage.get('winner', '-')}"
            )
    trials = state.get("trials", {})
    completed = sum(
        1 for trial in trials.values() if trial.get("status") in {"completed", "reused"}
    )
    print(f"参数试验记录: {completed}/{len(trials)}已完成或复用")
    if state.get("provisional_winner"):
        print(f"暂定冠军: {state['provisional_winner']}")
    if state.get("final_winner"):
        print(f"最终冠军: {state['final_winner']}")
    if state.get("last_error"):
        print(f"最近错误: {state['last_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
