# IPGDO 语义损失、缺陷锚点与引导强度参数搜索

## 搜索目标

本脚本只搜索以下三个 IPGDO 扩散参数：

- `formula_semantic_weight`
- `defect_anchor_start_fraction`
- `guidance_scale`

`target_minority_ratio=0.25`、`candidate_multiplier=3.5` 和
`physics_warmup_fraction=0.2`来自上一轮参数实验，本轮搜索期间固定不变。

## 参数选择协议

参数选择只使用 `source_cv`，不读取正式 CPDP 目标项目的评价结果。每个候选组合
使用相同的源项目分层三折、固定随机种子和逻辑回归分类器。过采样器只接触外层
训练折，分类阈值只使用训练折内部划分出的验证数据，六项指标最终在外层未见
测试折上计算。

每个候选先对同一项目的折和种子取均值，再按照 AUC、AUPRC、MCC、F1、
G-mean 和 Popt@20% 六项指标计算跨项目平均排名。AUC、AUPRC 和 Popt@20%
相对初始组合设有下降保护。正式 CPDP 结果不参与参数排名。

## 搜索流程

1. 搜索五个 `formula_semantic_weight`，保留前两名。
2. 固定第一阶段冠军，搜索五个 `defect_anchor_start_fraction`，保留前两名。
3. 固定前两阶段冠军，搜索五个 `guidance_scale`，保留前两名。
4. 对三个阶段的前两名执行 `2x2x2` 局部组合检查。
5. 最终冠军在五个筛选种子基础上补跑五个确认种子，总计十个种子。

不同阶段遇到相同参数组合时会复用完整结果。任务中断后重新运行同一命令即可
从未完成任务继续，不会重跑指纹一致且状态完整的结果。

## 使用方法

先验证配置和导入：

```powershell
python run_ipgdo_parameter_search.py --check
```

只生成任务计划，不启动训练：

```powershell
python run_ipgdo_parameter_search.py --dry-run
```

启动或继续参数搜索：

```powershell
python run_ipgdo_parameter_search.py
```

临时覆盖并发任务数：

```powershell
python run_ipgdo_parameter_search.py --workers 2
```

候选范围、种子和弹性 GPU 调度阈值统一配置在
`configs/ipgdo_parameter_search.yaml`。修改搜索范围后，搜索签名会变化，旧状态
不会被误认为当前搜索结果。

## 输出文件

新版结果保存在 `outputs/ipgdo_semantic_anchor_guidance_search`，不会覆盖上一轮
三个固定参数的历史结果。主要文件包括：

- `search_plan.yaml`：预计组合数和任务数。
- `search_state.json`：断点续跑状态。
- `stage_01_formula_semantic_weight_ranking.csv`：第一阶段排名。
- `stage_02_defect_anchor_start_fraction_ranking.csv`：第二阶段排名。
- `stage_03_guidance_scale_ranking.csv`：第三阶段排名。
- `stage_04_interaction_2x2x2_ranking.csv`：局部组合排名。
- `best_source_cv_project_metrics_10_seeds.csv`：冠军十种子逐项目均值。
- `best_params.yaml`：最终参数和可直接应用的配置覆盖项。
