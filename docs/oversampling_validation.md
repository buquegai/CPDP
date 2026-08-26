# 过采样模块独立验证框架

## 1. 与正式实验的边界

该框架位于 `src/cpdp/oversampling_validation/`，配置文件为
`configs/oversampling_validation.yaml`，结果保存到
`outputs/oversampling_validation/`。它不会向正式方法注册表添加方法，也不会写入正式
CPDP 实验的 `outputs/runs/`、`outputs/predictions/` 和正式汇总表。

`full` 变体每次运行都会实例化当前版本的
`IPGDOOversampler`。因此，后续修改扩散网络、隐性物理约束、数值有效性
筛选或 KNN 筛选后，验证代码会自动评估新版本，而不是保留某个旧模型副本。

## 2. 验证问题

框架分别回答两个问题：

1. `source_cv`：合成样本能否提高源项目内部未见真实样本上的分类性能？
2. `cpdp`：在统一的多源 LOPO 协议下，合成样本能否提高目标项目预测性能？

两种协议都比较同一划分、同一种子和同一固定逻辑回归分类器下的
`real_only` 与 `real_plus_synthetic`。主要结果是六项指标的配对差值：

`delta_metric = augmented_metric - baseline_metric`

## 3. 防止数据泄漏

`source_cv` 先保留外层真实测试折，再从外层训练部分划分真实验证集。过采样器、扩散缩放器、
物理公式和 KNN 阈值只接触内部训练部分；分类阈值只使用真实源验证集；最终指标只在外层真实
测试折计算。

`cpdp` 只从源项目划分训练集和阈值验证集。目标项目及其标签不会传给过采样器、分类器训练或
阈值选择，标签只在最终 `evaluate_predictions` 中使用。

## 4. 默认对照变体

- `none`：不做过采样，作为流程校验用的零增益变体。
- `random`：随机复制真实缺陷样本。
- `smote`：按项目执行自适应邻居数的 SMOTE。
- `tabddpm`：关闭物理损失、项目支持投影和 KNN 接受区间后的当前扩散管线。
- `full`：当前 IPGDO 管线。

新增消融实验只需在 YAML 的 `variants` 中添加一个 `kind: current` 变体，并通过
`overrides` 深度覆盖过采样配置。策略层只依赖
`fit_resample(sources, seed) -> AugmentationResult`，也可以通过
`register_strategy()` 注册全新的过采样器。

## 5. 质量指标

质量评估以训练折生成的缺陷样本和未参与生成器拟合的真实缺陷样本为对象，输出：归一化逐特征
Wasserstein 距离、RBF-MMD、相关矩阵差、精确/近重复率、KNN 缺陷流形距离、流形正常区间
比例、未见缺陷覆盖率和缺陷邻域纯度。生成器元数据中的数值有效性、物理公式和 KNN 诊断仍
完整保存在每个原始 JSON 中。

没有生成样本或未见折中缺陷样本不足时，相应指标写为缺失值并记录原因，不会伪造为 0。

## 6. 运行方式

快速运行可直接执行：

```powershell
python launch_oversampling_validation.py
```

脚本顶部可直接调整配置档、协议、变体、数据集、目标项目、种子和最大任务数。命令行示例：

```powershell
python run_oversampling_validation.py list --profile smoke
python run_oversampling_validation.py run --profile smoke --protocol source_cv --family NASA
python run_oversampling_validation.py run --profile full --protocol cpdp --variant full --target NASA/CM1
python run_oversampling_validation.py summarize --profile full
```

## 7. 结果目录与版本

结果路径格式为：

`outputs/oversampling_validation/<fingerprint>/<profile>/`

指纹同时覆盖有效验证配置、有效过采样配置、过采样源代码、验证分类器代码和数据文件内容。
任意一项发生变化都会生成新目录，因此旧结果不能被误认为新模型结果。同一指纹下已成功任务
会自动跳过；只有显式使用 `--force` 才会覆盖。

每个版本包含原始 JSON 和以下汇总：

- `summary/utility_runs.csv`：每个任务的基线、增强结果和六项增益。
- `summary/quality_runs.csv`：每个源项目的合成样本质量指标。
- `summary/project_utility_runs.csv`：每个被评估项目上的基线、增强结果和配对增益。
- `summary/utility_summary.csv`：先按种子聚合，再报告均值、中位数、标准差、胜率、配对
  Wilcoxon 检验和配对秩二列效应量。
- `summary/manifest.json`：当前指纹和完整运行数量。
