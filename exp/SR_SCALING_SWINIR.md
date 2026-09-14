# sr-scaling 的 SwinIR：他们改了什么、我们能学什么

**来源：** `third_party/sr-scaling/sr-scaling/sisr_scaling/arch_swinir.py` + Muon 训练主线

---

## 1. 结构上是不是「改过的 SwinIR」？

**基本是忠实复现的官方 SwinIR**，不是新架构：

| 模块 | 实现 |
|------|------|
| WindowAttention | 标准 shifted window + relative bias |
| RSTB | `x + conv(layer(x))`，支持 **1conv / 3conv** 残差连接 |
| 上采样 | 多级 **PixelShuffle(2)+LeakyReLU**（×2/4/8）或一次 ×3 |
| 正文 | 4 个 RSTB 组（`depths=[12,12,12,12]` 时为 p16m） |

**真正不同的在训练侧，不在注意力公式。**

---

## 2. 他们对 SwinIR 线的实质改动

| 改动 | 内容 | 对我们 |
|------|------|--------|
| **Muon 优化器** | 矩阵权重 Newton–Schulz；官方回归主线 SwinIR 用 `lr=1e-3` | **已移植** `model/optim.py` |
| **参数量分档** | p1m / p4m / p16m（如 p16m: embed=176, 4×12 blocks） | 可抄 bin 思路；8GB 上我们卡在 **~4–12M** |
| **纯 L1 损失** | `--loss l1` 默认 | 与我们结论一致 |
| **DF2K ×4 bicubic** | 数据与任务不同 | RealSR 配对真实退化，**不能直接抄数字** |
| **3conv 残差** | RSTB 尾 3 层瓶颈卷积 | 可选省参，非涨点核心 |

**没有** OCA/GDFN/AMF/FiLM/Align —— 那是我们为 RealSR 加的。

---

## 3. 定量（他们 bicubic ×4, DF2K, Muon）

| Bin | SwinIR PSNR/SSIM |
|-----|------------------|
| p1m | 28.90 / — |
| p4m | 29.03 / — |
| p16m | **29.13 / 0.823** |

结论：在**合成 bicubic** 上，SwinIR 单调吃参数；与 RealSR 上我们看到的「4M≈0.6M」不同——真实配对更受 **对齐与数据** 限制。

---

## 4. 建议学 / 不建议抄

| 学 | 不抄 |
|----|------|
| Muon（已接） | 完整 DF2K×4 协议当 RealSR 主结果 |
| 多级 PixelShuffle+LReLU 上采样（可试） | 16M 级 p16m（8GB 装不下） |
| 参数 bin 系统（p1m/p4m）做缩放表 | 他们的 PSNR 当我们 SOTA 对照 |
| 官方 SwinIR 复现做 **干净 baseline** | 再堆我们已否定的频域头 |

---

## 5. 与我们最优配方的对比

| | sr-scaling SwinIR | 我们 E11 |
|--|-------------------|----------|
| 主干 | 官方 SwinIR | +OCA/GDFN/AMF/FiLM/DCN |
| 重建 | 绝对（官方） | **残差 bicubic** |
| 损失 | L1 | **Align-L1** |
| 优化 | **Muon** | AdamW（Muon 可选未 A/B） |
| 任务 | bicubic ×4 | **RealSR 真实对 ×2** |

**可落地的下一步：** E11 + `--optimizer muon --lr 1e-3` 做 A/B；或上采样改成多级 PixelShuffle+LReLU 对照。
