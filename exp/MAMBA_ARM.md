# Mamba 臂：stride-1 VSSM（设计、实现与 A/B）

> **状态（2026-09-20 01:30）**：本文件的早期 A/B 协议（b64×20000 / 并行双车道）已被
> `PLAN_12H.md` 的**串行**计划取代（用户指令：串行、单任务吃满 40G、不含 U-Net 臂）。
> 实现与坑位记录仍然有效；跑的配置见 PLAN_12H。

> 动机（对应 `STAGE_SUMMARY.md` 的三条实测）：stride-1 是最大杠杆（+0.38 dB）→ 本网络**全程不降采样**；
> 压平金字塔 −0.07 dB 说明 HR 尺度需要长程上下文 → 选择扫描 O(L) 让 stride-1 全局感受野第一次
> 便宜到可用（全分辨率 self-attention 的 n² 项曾吃掉 DiT 64% 算力）；N1/N2（坐标、路由）平局说明
> 「按内容自适应」必须内生于算子 → S6 的 Δ/B/C 由输入动态生成，就是原生选择机制。
>
> N3（对抗退化）仍在跑（明早出数）；本臂是**架构类**的正交赌注。

---

## 架构（MambaIR 式残差组；扫描尺度是关键工程决定）

```
x → stem 3×3(stride=ssm_scale) → [RG ×4] → LN → 3×3 → (×scale pixel-shuffle，零初始化) → + bicubic
RG = Σ(VSSBlock ×4) → 3×3（零初始化）→ + identity
VSSBlock = LN → dwconv 3×3 → SiLU → SS2D → 1×1（零初始化）→ + identity
```

所有残差贡献者零初始化 → **step 0 精确等于 bicubic 地板**（smoke 已验证）。

**SS2D 交叉扫描**：4 个扫描序（行/列主序 × 正反）沿 batch 维拼接一次前向，均值融合。

**扫描尺度（--ssm-scale，实测决定）**：
- `scale=1`（stride-1，全程 HR）：L=16384，实测 **~30 s/step**（16 块 @ b64）→ 20000 步 = **7 天，不可行**。
  这就是"为什么比 U-Net 贵那么多"的答案：扫描是沿序列串行、按 (L×d_inner×N) 状态访存的算子，
  有效算力 ~1–2 TFLOPS（访存受限），而 U-Net 的卷积是稠密 GEMM（10–19 TFLOPS）+ 金字塔把算力
  放在低分辨率。参数量不是原因（2.75M vs U-Net 18.67M）。
- `scale=2`（**采用**，即 MambaIR 本来的设计）：stride-2 stem → L=4096 → 头部 pixel-shuffle ×2 回 HR。
  实测 **0.88 s/step @ batch 32、4GB 显存**（官方核 + causal-conv1d fast path + bf16 + 逐块 grad-ckpt）。
  降采样丢的相位由零初始化的 shuffle 头在 HR 端补回；step0 不变量不破。

**参数量（实测，纠正预估）**：

| 配置 | 参数 |
|---|---|
| c128 4×4（用户规格） | **2.75M**（预估的 25–30M 有误：Mamba 块是投影型算子） |
| c256 4×4 | 10.48M |

## 双后端（同一套参数与接口，ckpt 可互载）

| 后端 | 条件 | 速度 |
|---|---|---|
| `mamba_ssm`（官方 fused CUDA 核） | 服务器 conda 装 nvcc 12.8 后源码编译成功 | 快（预期 ≤0.5 s/step @ b64） |
| `torch`（纯 PyTorch S6） | 无 nvcc / 编译失败时的**精确**回退 | 慢 ~5× |

纯 torch 扫描的数学：Mamba 递推是**实对角**的（A<0 → a=exp(ΔA)∈(0,1)），有精确并行形式
`h_t = P_t(h_0 + Σ b_j/P_j)`，P = exp(cumsum(log a))。工程上踩了两个坑（都已修并写进代码注释）：
1. **cumsum clamp 会冻结衰减记忆** → 后面的 b 以权重 1 累加，误差 O(Σb)。改为**短段重整化**。
2. **前向有限、反向溢出**：除法 b/P 的反向要算 −b/P²，P < 1e-19 时溢出 fp32 → NaN 梯度
   （smoke 实测抓到）。段长自适应 = `38/(DT_MAX·N)`（dt 限幅 0.1 = Mamba 初始化同款），
   N=16 → 段长 23，反向中间量 ≤ 1e33。
   扫描全程 fp32 + 关 autocast（bf16 cumsum 256 步丢尾数）。
   **精确性已对照朴素循环验证：rel 1.4e-07（= fp32 精度），含最坏衰减 1000 步长序列。**

## A/B 协议（`scripts/server/run_mamba_arm.sh`）

锚点：`s1_b64` 零预训练 Y **34.1083** / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416（18.67M）。

| 臂 | 配方 | 样本预算 |
|---|---|---|
| M1 `mamba_c128` | dim128 / 4RG×4 / N16 / E2，其余与锚点逐字相同 | 与锚点完全一致：**1.28M 样本（batch 64 × 20000，两后端统一）** |
| M2 `mamba_c256` | 同上，dim256（10.48M） | 同上（torch 回退且未 `RUN_C256=1` 时跳过） |

判定（纪律同 FFN 教训）：官方 100 对 + IQA 定生死，±0.05 dB 内按平局；
16 对 val 只看趋势。**M1/M2 任一裸跑破 34.11 即架构路线成立**；若 c256 明显高于 c128，
说明 mamba 类在 390 对上也吃容量，与 U-Net 同规律。

## 排队与状态

- **服务器已装上官方 CUDA 核**：conda nvcc 12.8 + 源码编译 mamba-ssm 2.3.2.post1
  （torch 2.10/cu128，GPU 前后向测试通过）→ A/B 走快路径。
- 等 `newarms` 会话（N3 对抗预训练链，ETA 明早 ~04:35）结束后自动拉起（tmux `mambaarm`）。
- 日志：`experiments/diffusion/logs/mb_*.log`、`mamba_arm.log`；结果：`experiments/diffusion/mamba/<臂>/eval_iqa/eval.json`。

## 已知边界（明说）

- Mamba 是序列依赖的，官方评估仍走 tile 64/pad 16（cross-fade 32px 抑制接缝）——与其它臂同协议；
  若 M 臂出数接近锚点，需补一个 pad 32 的子集敏感性检查。
- `torch` 回退的纯扫描在 4090 上约 1–2 s/step（b64），M1 全程 ~10h；官方核成功则 ~1.5h。
- flow / latent 不支持（scan 忽略 t，reg-only，启动即校验拒绝）。
