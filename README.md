# 统一 CPDP 实验框架

本项目用于在同一问题空间内重新运行五种 CPDP 对比方法、本文提出的方法及其 Transformer-JEPA 架构变体。统一实验协议由 `configs/experiment.yaml` 固定：4 个数据集族、25 个目标项目、同族多源转导式 LOPO、目标标签仅用于公共评估、30 个固定随机种子，以及 AUC、AUPRC、MCC、F1、G-mean、Popt@20% 六项指标。

## 方法边界

| CLI 名称 | 论文方法 | 实现性质 | 论文中应使用的名称 |
| --- | --- | --- | --- |
| `dssdpp` | DSSDPP | 官方 MATLAB 算法的 Python 等价移植，CFS 调用官方 Weka JAR | DSSDPP |
| `master` | MASTER | 根据论文公式独立复现；官方仓库未发布 MASTER 核心 | MASTER (reimplemented) |
| `fedcpdp` | FedCPDP | 官方 NumPy 复核后端 + 等价手工公式 Torch/CUDA 批量后端 | FedCPDP |
| `fd_hdp_u` | FD-HDP | 去除目标标签、目标 SMOTE 和目标标签调参后的无标签适配 | FD-HDP-U |
| `mbl_cpdp_sv` | MBL-CPDP | 用源项目验证替代目标标签双层目标的概念适配 | MBL-CPDP-SV |
| `ipgdo_kan_jepa` | 本文方法 | 隐性物理约束引导扩散过采样与转导式 KAN-JEPA 预测器 | IPGDO + KAN-JEPA |
| `ipgdo_transformer_jepa` | 架构变体 | 仅将 Context Encoder、Target Encoder 和 Predictor 替换为 Transformer | IPGDO + Transformer-JEPA |

不能把 `master` 描述为官方源码复现，也不能把 `fd_hdp_u` 和 `mbl_cpdp_sv` 的结果直接冠以原方法名称。完整依据见 `docs/provenance.md`。

## 环境

项目支持 Python 3.9-3.12。当前机器默认 `python` 是不兼容依赖约束的 3.14，因此应显式使用 3.12：

```powershell
cd D:\PythonCode\CPDP
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
```

当前机器的 `.venv` 已安装 `torch 2.6.0+cu124`，FD-HDP-U 的 `device: auto` 会自动使用 RTX 4070。重新创建环境后，如默认安装得到 CPU 版 PyTorch，使用以下命令切换到 CUDA 12.4 版本：

```powershell
.\.venv\Scripts\python -m pip install --force-reinstall --no-deps `
  torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

DSSDPP 的 CFS 还要求 `java` 可执行；官方 `weka.jar` 已归档在 `third_party/original`。

## 命令

不需要手写命令时，直接在 PyCharm 中运行根目录的 `launch_experiments.py`。该入口会自动切换到项目 `.venv`，不受 PyCharm 当前选择的旧解释器影响。图形启动器支持方法勾选、smoke/full 切换、目标项目、随机种子、分批任务数、并行任务数、停止实验、数据验证、自动汇总和打开结果目录。

```powershell
.\.venv\Scripts\python launch_experiments.py
```

```powershell
# 查看方法及复现性质
.\.venv\Scripts\python run_experiment.py list-methods

# 验证 25 个项目、同族 schema、LOC 列和清洗结果
.\.venv\Scripts\python run_experiment.py validate-data

# 单方法、单目标、单种子冒烟
.\.venv\Scripts\python run_experiment.py run --method dssdpp --profile smoke --target ReLink/Apache --seed 2026072201

# 五种对比方法、本文方法及架构变体批量运行；已有成功 JSON 会自动跳过
.\.venv\Scripts\python run_experiment.py run --method all --profile full

# 限制本次新执行的任务数，便于分批运行
.\.venv\Scripts\python run_experiment.py run --method all --profile full --max-runs 10

# 同时运行 4 个相互独立的方法-目标-种子任务
.\.venv\Scripts\python run_experiment.py run --method all --profile full --workers 4

# 生成每次运行、目标项目均值/标准差、数据族宏平均和 25 项目宏平均
.\.venv\Scripts\python run_experiment.py summarize
```

每次运行保存在 `outputs/runs/<profile>/<method>/<family>/<target>/seed_<seed>.json`。启用预测保存时，同步写入含标签、分数、工作量和原始行号的压缩 NPZ。smoke 与 full 在物理路径和汇总分组中完全隔离。失败任务也写 JSON 和完整 traceback；再次运行时只跳过同 profile 的成功任务。

`target_means.csv` 的 `complete` 只有在六项指标都达到配置中的 30 次运行时才为真；族级和总体宏平均同时给出目标数及最小/最大运行数。中间汇总允许生成，但不得将 `complete=false` 的结果用于论文表格。

`outputs/summary` 根目录的四个 CSV 是包含当前全部成功结果的“最新总表”，每次汇总会刷新。各方法独立结果同时写入 `outputs/summary/by_method/<profile>/<method>/`，文件名包含方法名；每次汇总的不可覆盖快照保存在 `outputs/summary/history/<UTC时间戳>/`。底层 JSON/NPZ 默认断点续跑且不会覆盖，只有主动勾选“强制覆盖成功结果”时才会重写相同方法、项目和种子的原始结果。

同时运行全部方法共有 `7 x 25 x 30 = 5250` 个方法-目标-种子任务。应先完成全部 smoke，再按方法或目标分批执行 full，不能用 smoke 结果进入论文统计。

## 计算设备与并行

- FD-HDP-U 使用 `device: auto`：CUDA 可用时走 GPU，否则回退 CPU；实际设备写入每次结果的 `method_metadata.compute_device`。
- IPGDO、KAN-JEPA 与 Transformer-JEPA 均使用 `device: auto`，当前机器会调用 RTX 4070；实际设备和 AMP 状态写入方法元数据。
- IPGDO + JEPA 方法在单张 CUDA GPU 上强制使用一个外层任务；即使启动器填写更高的并行数，runner 也会自动降为 1，避免多个完整训练进程竞争显存并触发驱动故障。
- FedCPDP 在单个外层任务运行时并行训练 3 个独立集成种子，仍调用官方 NumPy 核心。
- MBL-CPDP-SV 在单个外层任务运行时并行评估同一上层方案的下层候选，候选顺序和随机种子保持不变。
- MBL-CPDP-SV 的候选验证和最终重训在独立子进程中执行；full 配置分别使用 20 秒和 60 秒硬超时，超时候选被淘汰且计数写入结果元数据，防止单个 SVM/MLP 候选突破整体时间预算。
- `--workers` 控制外层方法-目标-种子任务的独立进程数。FedCPDP 使用单张 CUDA GPU 时，runner 自动把外层任务数降为 1，并按 `outer_seed_batch_size` 将多个外层种子及其三个内部模型合并为一个设备批次；每个种子的参数、Adam 状态、随机序列、阈值和结果文件仍然独立。NumPy 复核后端仍可使用 CPU 内部并行。
- DSSDPP 和 MASTER 保持原方法对应的 CPU 后端。长任务每 60 秒输出一次 `WORK ... elapsed=...` 心跳日志。

当前 28 线程 CPU 的图形启动器默认使用 4 个外层并行任务。单独运行一个 FedCPDP 或 MBL-CPDP-SV 任务时，有效外层进程数会自动降为 1，从而启用方法内部并行。

## 测试

```powershell
# 快速单元测试，不启动真实数据集上的 IPGDO 组合方法
.\.venv\Scripts\python -m pytest

# 五种对比方法、本文方法及架构变体各在 ReLink/Apache 上运行一次 smoke
.\.venv\Scripts\python -m pytest -m integration
```

## 本文方法

正式方法名为 `ipgdo_kan_jepa`，其中 IPGDO 的全称为 Iterative Project-Guided Diffusion Oversampling。其过采样模块采用项目条件表格 DDPM 生成候选，并通过真实缺陷 KNN 距离区间循环筛选高质量样本；该模块不接收目标项目，KAN-JEPA 预测器只读取目标项目的无标签特征：

```powershell
.\.venv\Scripts\python run_experiment.py run `
  --method ipgdo_kan_jepa `
  --profile smoke --target ReLink/Apache --seed 2026072201
```

完整数据契约、网络结构、损失函数和模型选择策略见 `docs/kan_jepa_predictor.md`。目标标签由 runner 在预测结束后回接并统一计算指标。

Transformer 架构变体使用独立方法名 `ipgdo_transformer_jepa`，结果不会覆盖 KAN-JEPA。其三个 Transformer 模块与运行方式见 `docs/transformer_jepa_predictor.md`。
