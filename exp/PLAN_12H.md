# 12 小时串行实验计划（2026-09-20 起）

> 纪律修正：**串行**，一次一个任务；**不含任何 U-Net 臂**（N3 微调维持取消状态，
> 其 ckpt_best 保留在盘上，问题悬置）；单任务用 batch 把显存顶到 ~40GB。
> 脚本：`scripts/server/run_12h_plan.sh`（单 tmux 会话 `plan12`，串行执行，幂等可断点重入）。

## 任务序列

| # | 臂 | 配置 | batch（探测定） | 样本预算 | 预计墙钟 |
|---|---|---|---|---|---|
| 0 | 探针 | 每个 batch 候选先跑 8 步实测显存，OOM 自动降档 | — | — | ~4 min |
| 1 | **M1 `mamba_c128`**（2.75M） | dim128 / 4RG×4 / N16 / E2 / scale-2 扫描，1600 步 | 实测：384 OOM（46.6GB）→ **320 通过（39.4GB）** | **512k** | ~6.4h + 评估 0.4h |
| 2 | **M2 `mamba_c256`**（10.48M） | 同上，base 256，1000 步 | 192→160→128→96 | **~192k** | ~4.8h + 评估 0.4h |

总 ETA ≈ **12h**。实测吞吐：c128 @ b320 ≈ 14.5 s/step（22 samples/s，batch 无关）；
c256 单样本成本 ≈ 2×（~11 samples/s）。大 batch 只为吃满显存，吞吐不亏。

## 判定基准（官方 100 对 + IQA，±0.05 dB 内按平局）

| 对照 | Y |
|---|---|
| 锚点 s1_b64 零预训练（1.28M 样本） | 34.1083 |
| 预训练线上限（BSRGAN） | 34.1988 |
| bicubic 地板 | 31.7335 |

## 计划回答的问题

1. **M1**：VSS/Mamba 类在 390 对 RealSR 上的样本效率 —— 768k 样本（锚点预算的 60%）
   若能到 ≥34.0，说明该类比 U-Net 更省数据，值得全预算复赛。
2. **M2**：容量拐点规律（U-Net 上 5M→18.7M 才涨）在 Mamba 上是否复现 —— M2/M1 参数差 3.8 倍。

## 配置要点（全部实测得出）

- 官方栈：mamba-ssm 2.3.2.post1 + causal-conv1d（nvcc 12.8 自编译，fast path 开启）。
- **scale-2 扫描**（stride-2 stem → L=4096 → pixel-shuffle 回 HR）：stride-1 的 L=16384
  实测 ~30s/step（7 天/臂）不可行；scale-2 实测 0.88s/step @ b32、4GB。
- 零初始化头 → step0 = bicubic 地板；`expandable_segments` 抗并行/碎片；
  大 batch 与 lr 3e-4（与锚点同）、warmup 500、EMA 0.999、cosine。
- 评估：与全仓库同协议（tile 64 / pad 16 / steps 8 + MUSIQ/MANIQA）。
