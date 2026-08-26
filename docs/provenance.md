# 方法来源与统一协议适配

## 可审计原则

所有方法都通过 `BaseMethod.fit_predict(sources, target, seed)` 接口运行。`sources` 含源项目标签，`target` 的类型是没有标签字段的 `UnlabeledProject`；其独立 `UnlabeledFeatures` 视图也不暴露原始 CSV 路径，避免方法绕过接口重新读取目标标签。公共 runner 只在方法返回分数后读取目标标签并计算六项指标。原始压缩包和关键入口文件的 SHA-256 记录在 `third_party/SHA256SUMS.txt`。

统一协议只统一数据集、外层同族多源 LOPO、目标标签可见性、随机种子、指标和汇总方式。SMOTE、MPOS、Focal Loss 等类别不平衡机制属于各方法算法核心，不做跨方法强制替换。

## DSSDPP

- 来源：Zenodo `https://zenodo.org/records/6651565`，官方 MATLAB 包 `DSSDPP_v3`。
- 保留：CORR 内部单源选择、MPOS、CFS/BestFirst、DPP 线性规划。
- 适配：MATLAB 数值流程移植到 NumPy/SciPy；CFS 直接调用包内官方 Weka JAR；外层候选源限制为同族其余项目。
- 协议差异：官方源码随机抽取 10% 无标签目标模块用于源选择；统一协议使用完整无标签目标特征，且保留 30 个外层固定种子。

## MASTER

- 来源：`https://github.com/THN-BUAA/MASTER-master`，快照 `2315e1a764617f0bc7e23041c33d121282e02cb9`。
- 关键事实：仓库 README 明确写明只包含数据集和基线，MASTER 核心“will be released soon”。
- 实现：依据论文公式 (1)-(15) 独立实现逐源 SMOTE、MDL 离散化、MI/KL 迁移权重、特征加权 TNB 和多源加权集成。
- MIC：优先使用 `minepy`；不可用时使用 scikit-learn 互信息并在每次结果元数据中明确记录 fallback。
- 报告要求：应写为 paper-derived reimplementation，不能写成基于官方 MASTER 源码。

## FedCPDP

- 来源：`https://github.com/MeeteshNevendra/FedCPDP`，归档快照 `ab5af63`。
- 保留：官方 NumPy MLP、本地训练、FedProx、Focal Loss/weighted BCE、表示对齐、自适应聚合、源数据阈值选择和多种子集成。
- 适配：输入改为统一内存项目对象；官方 Python `hash()` 受进程随机化影响，改用确定性 BLAKE2b-64 映射并记录在元数据中。
- 标签边界：阈值仅由源项目训练分数和源标签选择，目标标签不进入联邦训练或阈值搜索。

## FD-HDP-U

- 来源：Zenodo `https://doi.org/10.5281/zenodo.15332940`，记录 `15332941`，代码快照 `70a012f`。
- 原方法冲突：官方流程对目标数据执行 SMOTE，使用目标标签选择 `alpha`，并用目标标签微调和评估，属于带目标标签的异构/半监督设置。
- 无标签适配：只对合并源数据执行官方 80% SMOTE；使用无标签目标特征参与重构和域对抗；共享缺陷预测器只接受源标签监督；固定 `alpha_T=1.0`。
- 报告要求：统一实验中的方法名必须是 FD-HDP-U，不能声称是原版 FD-HDP 的原协议结果。

## MBL-CPDP-SV

- 来源：`https://github.com/JiaxinChen-Archive/MBL-CPDP`，归档快照 `ae49526`。
- 原方法冲突：官方 `Time20.py` 将 `Ytarget` 传入每个下层优化目标，无法在无目标标签协议下原样运行。
- 源验证适配：保留上层流水线选择、下层超参数搜索、双目标 Pareto 前沿和集成；候选性能改由留出源项目验证得到。
- 当前搜索空间：NNfilter/CDE-SMOTE、5 类特征选择和 14 类可稳定输出分数的分类器。它不是官方完整搜索空间的逐行移植。
- 计算保护：候选源验证和最终重训使用配置化的独立子进程硬超时；超时只淘汰当前候选，不改变候选生成顺序、参数范围、双目标选择或 Pareto 集成规则。每次结果记录搜索/重训超时次数和实际耗时。
- 报告要求：统一实验中的方法名必须是 MBL-CPDP-SV，并在威胁有效性中说明源验证替换。

## 数据审计备注

AEEEM 中包含 `numberOfBugsFoundUntil:` 等历史缺陷计数指标；抽查表明它们与统一 `defect_label` 并不等价，不能仅凭列名认定为直接标签副本并擅自删除。正式实验前仍应依据所采用 AEEEM 版本的数据字典说明这些历史指标与预测标签的时间边界，并将最终特征清单归档。

## 计算后端与并行边界

- DSSDPP 与 MASTER 保持 CPU 数值后端，不以 GPU 库替换 Weka、SciPy 或论文公式实现。
- FedCPDP 的多个官方种子模型可并行训练，但每个模型内部仍调用官方 NumPy MLP、FedProx、Focal Loss、表示对齐和聚合函数。
- FD-HDP-U 只增加 PyTorch `device` 迁移，网络结构、损失和训练轮数不变；结果元数据必须记录实际 CPU/CUDA 设备。
- MBL-CPDP-SV 只并行相互独立的下层候选评价，随机候选生成仍由主线程按固定顺序完成。
- 外层并行只调度相互独立的方法-目标-种子任务。启用外层多进程时关闭方法内部并行，防止嵌套并行改变资源竞争条件。
