# 归档文档（Archive）

这里的文档是**过程档案**，保留原文以便追溯当时怎么想的、试过什么。
它们被以下四份正式文档取代，**结论以正式文档和
[`docs/results/RESULT_LEDGER.md`](../../docs/results/RESULT_LEDGER.md) 为准**：

| 正式文档 | 管什么 |
|---|---|
| [`../../docs/results/RESULT_LEDGER.md`](../../docs/results/RESULT_LEDGER.md) | 全部实测数字（唯一底账） |
| [`../FINAL_REPORT.md`](../FINAL_REPORT.md) | 逐次实验的假设/配置/结论编年史 |
| [`../RELATED_WORK_AND_UPPER_BOUND.md`](../RELATED_WORK_AND_UPPER_BOUND.md) | 相关工作与物理上限 |
| [`../METRIC_SUITE_AND_DCP.md`](../METRIC_SUITE_AND_DCP.md) | 评测体系与 WD-DCP 理论 |

## 为什么不删

这些文档里有正式文档**故意没收录**的东西：失败的中间尝试、当时的错误判断、
被推翻的假设。写论文与复盘时，这些是最容易再踩一次的地方。

## 清单

### 阶段一：SwinIR（09-13 ~ 09-14）
| 文件 | 内容 | 与今天的结论的关系 |
|---|---|---|
| `SR_SCALING_SWINIR.md` | sr-scaling 的 SwinIR 改动与我们官方版的差异 | 历史，已并入正式文档 |
| `SR_SCALING_PORT.md` | 移植清单 | 同上 |
| `SWINIR_LINE.md` | SwinIR 线的路线图 | 已被 U-Net 取代 |
| `SWINIR_DATA_PROTOCOL.md` | 为什么 E11 仍是公平锚点 | **方法学仍然有效** |
| `REGRESSION_LINE.md` | 逐改动指标增量 | 方法论样本 |
| `VERIFIED_SUMMARY.md` | 只保留正收益的配方汇总 | 已被 champion 取代 |

### 阶段二：扩散（09-16 ~ 09-17）
| 文件 | 内容 |
|---|---|
| `DIFFUSION_PLAN.md` | 2×2 矩阵计划 |
| `DIT_LINE.md` | DiT 线的起步 |
| `PIXEL_DIFFUSION_MEMORY.md` | 像素扩散的显存/吞吐实测 |
| `VAE_NOISE_FLOOR_REALSR.md` | Flux VAE 往返噪声底 44.7 dB（later 实测 31.5971） |
| `EVAL_PROTOCOL.md` | 早期评测协议定义（仍是现行协议来源） |
| `BASELINE_PROTOCOL.md` | 早期基线协议约定 |
| `ABLATION_FULL.md` | 早期消融表 |

### 阶段三：Mamba（09-20）
| 文件 | 内容 |
|---|---|
| `MAMBA_ARM.md` | 单个 Mamba 臂的设计 |
| `MAMBA_MATRIX_RESULTS.md` | 8 臂矩阵的结果（线关闭） |

### 阶段四：小波与优化器（09-22 ~ 09-25）
| 文件 | 内容 |
|---|---|
| `WAVE_ARMS.md` | 小波臂第一轮 |
| `NEW_ARMS.md` | CoordConv / 频域路由 / 对抗退化 |
| `STAGE_SUMMARY.md` | 阶段小结 |
| `PROGRESS_REPORT.md` | 完整进度报告 |
| `RESULTS_SUMMARY.md` | 阶段结果汇总 |
| `MODEL_COMPARISON.md` | 模型横评 |
| `PLAN_12H.md` | 12 小时排程计划 |
| `TRAIN_HARDENING.md` | 训练稳定性加固 |
| `NEGATIVE.md` | 早期负结果清单 |
| `RESULTS.md` | 早期结果表 |
| `IDEAS_OT_AND_BEYOND.md` | OT 损失的设想（后被证伪） |

### 上游资料
| 文件 | 内容 |
|---|---|
| `README.upstream.md` | 上游仓库（KAIR / RealSR 工具链）的 README 原文 |
| `OFFICIAL_EVAL_PROGRESS.md` | 早期官方评估进度 |
