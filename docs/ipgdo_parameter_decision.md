# IPGDO 固定参数决策记录

## 决策

截至 2026-08-08，四轮参数搜索完成后，后续 IPGDO 消融实验和其他参数实验固定以下环境：

| 参数 | 固定值 | 含义 |
| --- | ---: | --- |
| `target_minority_ratio` | `0.6` | 每个源项目最终缺陷样本数达到正常样本数的 60% |
| `candidate_multiplier` | `3.5` | 最终需要一个合成样本时，预先生成 3.5 个候选样本 |
| `physics_warmup_fraction` | `0.2` | 前 20% 训练轮次只优化扩散损失，之后启用物理损失 |
| `formula_semantic_weight` | `0.05` | 公式语义物理损失权重 |
| `defect_anchor_start_fraction` | `0.5` | 从扩散过程 50% 位置开始使用真实缺陷锚点 |
| `guidance_scale` | `1.5` | 分类器自由引导强度 |
| `physics_timestep_fraction` | `0.4` | 物理损失参与训练的扩散时间步比例 |
| `condition_dropout` | `0.1` | 条件分支训练时的条件丢弃概率 |
| `manifold.lower_quantile` | `0.025` | 真实缺陷流形接收区间下界 |
| `manifold.upper_quantile` | `0.975` | 真实缺陷流形接收区间上界 |
| `manifold.neighbors` | `5` | 流形距离估计的近邻数 |
| `formula.max_terms` | `10` | 隐式物理公式最多保留的非零项数 |

正式配置位于 `configs/experiment.yaml` 的
`profiles.full.ipgdo_kan_jepa.oversampling`。Transformer-JEPA 架构变体通过
YAML 锚点使用完全相同的 IPGDO 配置。过采样独立验证框架在
`configs/oversampling_validation.yaml` 的 `variants.full.overrides` 中再次显式
固定完整参数组合，以避免后续编辑产生隐式漂移。

`smoke` profile 保持相同的算法参数，但缩减网络规模、扩散步数、训练轮数、
生成上限和候选池，只用于检查程序能否运行，不得进入论文统计。

## 第一轮选择依据（历史记录）

参数搜索覆盖 25 个目标项目和 AUC、AUPRC、MCC、F1、G-mean、Popt@20%
六项指标。所有候选先使用 5 个固定种子筛选，最终候选再补充 5 个种子，合计
10 个种子。候选按照 150 个“目标项目 × 指标”单元格的平均排名选择，并对
AUC、AUPRC 和 Popt@20% 设置相对初始配置的下降保护。

第一轮最终 10 种子宏平均结果为：

| AUC | AUPRC | MCC | F1 | G-mean | Popt@20% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.728516 | 0.474768 | 0.263399 | 0.413429 | 0.598557 | 0.418878 |

可审计来源：

- `outputs/ipgdo_parameter_search/best_params.yaml`
- `outputs/ipgdo_parameter_search/best_target_project_metrics_10_seeds.csv`
- `outputs/ipgdo_parameter_search/stage_04_interaction_2x2x2_ranking.csv`

## 结果隔离

正式 KAN-JEPA 和 Transformer-JEPA 配置分别使用独立的
`result_signature`。旧参数生成的成功 JSON 不满足新签名，因此断点续跑不会
错误地跳过新参数任务。修改任何已固定参数时，必须同步更新签名并建立新的决策
记录，不能覆盖本次参数方案的实验含义。

过采样对照实验中的 ROS、SMOTE 和 TabDDPM 继续显式使用此前的对照设置，
不会继承 IPGDO 的专用参数组合。

## 论文阶段注意事项

本次搜索用于模型开发阶段。由于候选排名使用了目标项目评估指标，正式论文在
报告无偏泛化结论前，应采用独立开发项目、嵌套项目级验证或源项目内部验证重新
确认固定参数，避免在最终测试项目上选择超参数造成乐观偏差。

## 2026-08-06 Source-CV确认

为消除上述目标项目调参偏差，第二轮搜索只使用源项目内部三折交叉验证。所有
候选先运行5个固定种子，三个阶段各保留前两名，再执行`2x2x2`局部组合检查；
冠军补充到10个种子。最终固定：

| 参数 | 固定值 |
| --- | ---: |
| `formula_semantic_weight` | `0.05` |
| `defect_anchor_start_fraction` | `0.5` |
| `guidance_scale` | `1.5` |

可审计来源：

- `outputs/ipgdo_semantic_anchor_guidance_search/best_params.yaml`
- `outputs/ipgdo_semantic_anchor_guidance_search/best_source_cv_project_metrics_10_seeds.csv`
- `outputs/ipgdo_semantic_anchor_guidance_search/stage_04_interaction_2x2x2_ranking.csv`

上述参数已经写入正式配置。Source-CV结果只用于选择参数，不得直接与CPDP多源
LOPO结果混合；参数冻结后必须另行运行正式CPDP协议，才能更新目标项目对比表。

## 2026-08-07 第三轮Source-CV搜索

`target_minority_ratio=0.25`重新开放为待搜索参数，并作为本轮搜索起点。第三轮
联合考察`target_minority_ratio`、
`physics_timestep_fraction`和`condition_dropout`，继续采用5种子初筛、三个阶段
各保留前两名、`2x2x2`局部交互检查、冠军补足至10种子的Source-CV协议。

搜索已完成，最终固定：

| 参数 | 固定值 |
| --- | ---: |
| `target_minority_ratio` | `0.6` |
| `physics_timestep_fraction` | `0.4` |
| `condition_dropout` | `0.1` |

本轮结果与历史搜索隔离保存于：

- `outputs/ipgdo_ratio_timestep_condition_search/`
- `outputs/ipgdo_ratio_timestep_condition_search/validation_runs/`
- `outputs/ipgdo_ratio_timestep_condition_search/best_params.yaml`
- `outputs/ipgdo_ratio_timestep_condition_search/best_source_cv_project_metrics_10_seeds.csv`
- `outputs/ipgdo_ratio_timestep_condition_search/ipgdo_three_parameter_sensitivity.png`

Source-CV冠军仍需经过正式CPDP协议复验，复验结果才可写入目标项目对比表。

## 2026-08-08 第四轮Source-CV搜索

第四轮联合考察流形分位区间、流形邻居数和公式最大项数。流形上下分位数作为
一个参数组共同变化，继续采用5种子初筛、各阶段保留前两名、`2x2x2`局部交互
检查、冠军补足至10种子的Source-CV协议。最终固定：

| 参数 | 固定值 |
| --- | ---: |
| `manifold.lower_quantile` | `0.025` |
| `manifold.upper_quantile` | `0.975` |
| `manifold.neighbors` | `5` |
| `formula.max_terms` | `10` |

冠军随后完成25个目标项目、10个随机种子的正式CPDP复验，宏平均结果为：

| AUC | AUPRC | MCC | F1 | G-mean | Popt@20% |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.728031 | 0.471911 | 0.265922 | 0.414732 | 0.597218 | 0.414803 |

可审计来源：

- `outputs/ipgdo_manifold_formula_search/best_params.yaml`
- `outputs/ipgdo_manifold_formula_search/best_source_cv_project_metrics_10_seeds.csv`
- `outputs/ipgdo_manifold_formula_search/best_cpdp_target_means_10_seeds.csv`

## 统一生效范围

当前十二个既定参数统一应用于正式KAN-JEPA、正式Transformer-JEPA和独立IPGDO
过采样验证。冒烟配置中的
`formula_semantic_weight`、`defect_anchor_start_fraction` 与 `guidance_scale` 也与
正式配置保持一致；冒烟测试仍只缩减网络规模、扩散步数、训练轮数和生成上限，
因此不得用于论文结果。

ROS、SMOTE 和纯 TabDDPM 是独立对照方法，不继承 IPGDO 固定参数。任何为验证
机制贡献而显式置零或关闭参数的消融变体也继续保留其覆盖值。

## 后续调参优先级

前四轮已经确定上述十二个参数。后续再依次考察：

1. `formula.l1_c`：公式稀疏性与拟合能力之间的权衡。
2. `diffusion.learning_rate`：扩散网络优化步长。
3. `diffusion.epochs`：扩散网络训练预算。

网络容量参数 `hidden_dim`、`residual_blocks` 与 `timesteps` 放在上述参数稳定后再
联合检查。`batch_size`、`sample_batch_size`、`use_amp` 和
`diversity_pool_limit` 主要用于资源与时间控制，除非出现收敛或候选池截断问题，
不作为首轮性能调参对象。
