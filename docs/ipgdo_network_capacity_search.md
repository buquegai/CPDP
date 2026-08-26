# IPGDO扩散网络容量参数搜索

## 搜索参数

本轮只搜索以下三个参数：

| 参数 | 候选值 |
| --- | --- |
| `diffusion.hidden_dim` | `128, 192, 256, 384, 512` |
| `diffusion.residual_blocks` | `2, 3, 4, 5, 6` |
| `diffusion.timesteps` | `100, 150, 200, 300, 400` |

`hidden_dim` 控制去噪网络内部表示宽度，`residual_blocks` 控制网络深度，二者共同决定模型容量与单步计算成本。`timesteps` 控制扩散离散过程的精细程度，并直接影响反向去噪的迭代次数。参数变大不保证生成质量单调提升，因此仍以统一Source-CV结果选择冠军。

本轮以第五轮冠军为固定环境：`formula.l1_c=0.1`、`diffusion.learning_rate=0.001`、`diffusion.epochs=300`，其余已确定参数也全部保持不变。质量评价中的流形分位阈值固定为 `0.05-0.95`，不随模型流形参数变化。

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
python run_ipgdo_parameter_search.py --check
```

生成计划但不启动训练：

```powershell
python run_ipgdo_parameter_search.py --dry-run
```

正式启动或断点续跑：

```powershell
python run_ipgdo_parameter_search.py
```

本轮结果保存在 `outputs/ipgdo_network_capacity_search/`。第五轮仍可通过 `run_ipgdo_formula_optimization_search.py` 独立复现，其配置和历史结果目录不会被覆盖。
