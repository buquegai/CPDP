# IPGDO网络容量参数搜索Linux包

本包用于在Linux云服务器上运行当前一轮IPGDO参数实验，搜索：

- `diffusion.hidden_dim`
- `diffusion.residual_blocks`
- `diffusion.timesteps`

包内包含当前源码、25个统一CSV项目、参数搜索配置、断点续跑逻辑和10种子冠军完整CPDP复验。未包含本地历史输出、虚拟环境和缓存。

## 1. 当前固定的最佳参数

以下参数已经按前五轮Source-CV实验结果固定：

| 参数 | 当前值 |
| --- | ---: |
| `target_minority_ratio` | `0.6` |
| `candidate_multiplier` | `3.5` |
| `formula.l1_c` | `0.1` |
| `formula.max_terms` | `10` |
| `diffusion.epochs` | `300` |
| `diffusion.learning_rate` | `0.001` |
| `diffusion.physics_warmup_fraction` | `0.2` |
| `diffusion.formula_consistency_weight` | `0.0` |
| `diffusion.formula_semantic_weight` | `0.05` |
| `diffusion.defect_anchor_start_fraction` | `0.5` |
| `diffusion.guidance_scale` | `1.5` |
| `diffusion.physics_timestep_fraction` | `0.4` |
| `diffusion.condition_dropout` | `0.1` |
| `manifold.lower_quantile / upper_quantile` | `0.025 / 0.975` |
| `manifold.neighbors` | `5` |

本轮三个待搜索参数的起点为 `hidden_dim=256`、`residual_blocks=4`、`timesteps=200`。它们是当前基线，不应在本轮结束前写成新冠军。

## 2. 上传与解压

```bash
unzip IPGDO_NetworkCapacity_Linux_Package_20260809.zip
cd IPGDO_NetworkCapacity_Linux_Package_20260809
```

## 3. 安装环境

默认安装PyTorch 2.6.0 CUDA 12.4版本：

```bash
bash setup_linux.sh cu124
```

也可根据服务器环境选择：

```bash
bash setup_linux.sh cu118
bash setup_linux.sh cu126
bash setup_linux.sh cpu
```

要求Python版本为3.9-3.12。完整实验建议使用CUDA，不建议使用CPU运行。

## 4. 正式运行前检查

```bash
bash check.sh
bash plan.sh
```

`check.sh`会检查25个数据文件、固定冠军参数、Python依赖和CUDA状态。`plan.sh`只生成任务计划，不启动训练。

## 5. 启动实验

前台运行：

```bash
bash run_search.sh
```

后台运行：

```bash
bash run_detached.sh
```

查看状态：

```bash
bash status.sh
tail -f logs/network_capacity_search.nohup.log
```

实验被中断后，重新执行同一个启动命令即可断点续跑；已完成且指纹一致的任务会自动复用。

## 6. 调整参数

主要配置文件为：

```text
configs/ipgdo_network_capacity_search.yaml
```

可直接修改：

- `parameter_grids`：三个参数的候选值。
- `seeds.screening`和`seeds.confirmation`：筛选与确认种子。
- `execution.workers`：并行任务数，默认4。
- `execution.elastic_gpu`：GPU利用率、空闲显存和温度门槛。

前五轮冠军值保存在`fixed_overrides`中。本轮实验期间不要随意修改这些值，否则新的结果不再处于同一固定实验环境。

## 7. 输出位置

所有新结果保存到：

```text
outputs/ipgdo_network_capacity_search/
```

主要文件包括：

- `search_state.json`：断点状态。
- `stage_*_ranking.csv`：各阶段候选排名。
- `best_params.yaml`：最终冠军参数及结果路径。
- `best_source_cv_project_metrics_10_seeds.csv`：冠军10种子Source-CV结果。
- `best_cpdp_target_means_10_seeds.csv`：冠军25个目标项目的10种子均值。
- `ipgdo_network_capacity_sensitivity.png/.pdf`：参数敏感性图。

本轮最多20组唯一参数配置，任务计划上限为1200个筛选任务、60个确认任务和250个最终CPDP任务，共1510个。
