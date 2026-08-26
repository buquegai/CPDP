# IPGDO 第四轮参数搜索与冠军CPDP复验

## 研究目标

本轮在前三轮已确定参数的基础上，只搜索以下三个参数组：

1. `manifold.lower_quantile / upper_quantile`：作为一个成对参数共同变化，控制真实缺陷内部距离的正常接收区间。
2. `manifold.neighbors`：控制流形距离估计使用的近邻数量。
3. `formula.max_terms`：控制隐式物理公式最多保留的非零项数。

流形上下分位数必须成对搜索。若分别独立排列，不但会产生大量组合，还可能产生
`lower_quantile >= upper_quantile` 的无效配置，因此本轮不采用两个参数的笛卡尔积。

## 候选范围

| 参数组 | 候选值 |
| --- | --- |
| 流形分位区间 | `0.01-0.99`、`0.025-0.975`、`0.05-0.95`、`0.10-0.90`、`0.15-0.85` |
| `manifold.neighbors` | `3`、`5`、`7`、`9`、`11` |
| `formula.max_terms` | `4`、`6`、`8`、`10`、`12` |

本轮实际搜索起点为当时的正式设置：流形区间 `0.05-0.95`、邻居数 `5`、公式最大项数
`10`。质量评价模块中的流形阈值固定不变，避免评价标准跟随待搜索参数变化。

## 固定基线

搜索期间始终固定前三轮得到的八个最佳参数：

| 参数 | 固定值 |
| --- | ---: |
| `target_minority_ratio` | `0.6` |
| `candidate_multiplier` | `3.5` |
| `physics_warmup_fraction` | `0.2` |
| `formula_semantic_weight` | `0.05` |
| `defect_anchor_start_fraction` | `0.5` |
| `guidance_scale` | `1.5` |
| `physics_timestep_fraction` | `0.4` |
| `condition_dropout` | `0.1` |

公式可靠性软加权继续关闭，`formula_consistency_weight` 继续保持为 `0.0`。

## 搜索协议

1. 三个阶段分别搜索流形区间、邻居数和公式项数。
2. 每个阶段使用5个固定种子，并保留综合平均排名前两名。
3. 对三个阶段的前两名执行 `2×2×2` 局部组合检查。
4. 临时冠军补充后5个种子，使Source-CV确认结果达到10个种子。
5. 参数选择只使用源项目内部三折Source-CV，不读取CPDP目标项目结果。
6. 冠军确定后，自动运行25个目标项目、10个种子的完整CPDP复验。

最多产生20个唯一搜索组合，对应1200个筛选任务和60个冠军确认任务。自动
CPDP复验包含250个任务，因此整个流程的理论任务上限为1510。重复组合和已经
成功完成的任务会自动跳过。

## 运行方法

检查配置但不创建结果：

```powershell
python run_ipgdo_parameter_search.py --check
```

生成任务计划但不启动训练：

```powershell
python run_ipgdo_parameter_search.py --dry-run
```

正式运行：

```powershell
python run_ipgdo_parameter_search.py
```

实验中断后再次运行同一脚本即可断点续跑。旧的第三轮
`target_minority_ratio / physics_timestep_fraction / condition_dropout` 搜索可通过
`run_ipgdo_ratio_timestep_condition_search.py` 复现，不会与第四轮结果混合。

## 最终结果

第四轮 Source-CV 冠军已经固定为：

| 参数 | 最优值 |
| --- | ---: |
| `manifold.lower_quantile` | `0.025` |
| `manifold.upper_quantile` | `0.975` |
| `manifold.neighbors` | `5` |
| `formula.max_terms` | `10` |

该组合已完成 25 个目标项目、10 个随机种子的正式 CPDP 复验。

第四轮所有文件保存在 `outputs/ipgdo_manifold_formula_search/`：

| 文件 | 用途 |
| --- | --- |
| `best_params.yaml` | 最终参数、Source-CV指标、CPDP宏平均和全部结果路径 |
| `best_source_cv_project_metrics_10_seeds.csv` | 冠军的10种子Source-CV项目级结果 |
| `best_cpdp_runs_10_seeds.csv` | 25项目×10种子的CPDP逐次结果 |
| `best_cpdp_target_means_10_seeds.csv` | 每个目标项目一行的六指标10种子均值，可直接用于更新过采样对比表 |
| `ipgdo_manifold_formula_sensitivity.png` | 三个参数组的论文式敏感性图 |
| `search_state.json` | 搜索和最终CPDP复验的断点状态 |

最终CPDP采用与过采样方法对比实验一致的 Logistic Regression 评价器。其目的
是隔离并衡量过采样模块贡献，不是替代完整的 KAN-JEPA 预测模型实验。
