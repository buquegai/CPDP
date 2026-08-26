# IPGDO第五轮参数搜索

## 搜索参数

本轮只搜索以下三个参数：

| 参数 | 候选值 |
| --- | --- |
| `formula.l1_c` | `0.025, 0.05, 0.1, 0.2, 0.5` |
| `diffusion.learning_rate` | `0.0001, 0.00025, 0.0005, 0.00075, 0.001` |
| `diffusion.epochs` | `150, 225, 300, 400, 500` |

`formula.l1_c` 是L1逻辑回归正则强度的倒数：值越小，稀疏约束越强；值越大，保留的公式项通常越多。`diffusion.epochs` 同时影响收敛程度与运行时间，因此仅在指标相同的情况下优先选择训练轮数更少的候选。

前三轮确定的八项参数以及第四轮确定的流形区间、邻居数和公式项数均保持固定。质量评价中的流形分位阈值继续固定为 `0.05-0.95`，不会随模型流形参数变化。

## 搜索协议

1. 三个参数依次进行单参数搜索，每阶段保留综合平均排名前两名。
2. 对三个阶段前两名执行 `2×2×2` 局部交互检查。
3. 筛选阶段使用5个种子和三折Source-CV，冠军再补充5个确认种子。
4. 参数选择不读取CPDP目标项目结果。
5. 冠军确定后自动执行25个目标项目、10个种子的完整CPDP复验。
6. 默认4并发，保留CUDA瞬时异常重试和断点续跑。

## 运行方式

仅检查配置：

```powershell
python run_ipgdo_formula_optimization_search.py --check
```

生成计划但不启动训练：

```powershell
python run_ipgdo_formula_optimization_search.py --dry-run
```

正式启动或断点续跑：

```powershell
python run_ipgdo_formula_optimization_search.py
```

本轮结果保存在 `outputs/ipgdo_formula_optimization_search/`。第五轮由 `run_ipgdo_formula_optimization_search.py` 独立复现，当前主入口 `run_ipgdo_parameter_search.py` 已用于后续网络容量参数搜索；历史结果目录不会被覆盖。
