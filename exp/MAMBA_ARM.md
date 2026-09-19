# Mamba 臂：stride-1 VSSM（设计、实现与 A/B）

> 动机（对应 `STAGE_SUMMARY.md` 的三条实测）：stride-1 是最大杠杆（+0.38 dB）→ 本网络**全程不降采样**；
> 压平金字塔 −0.07 dB 说明 HR 尺度需要长程上下文 → 选择扫描 O(L) 让 stride-1 全局感受野第一次
> 便宜到可用（全分辨率 self-attention 的 n² 项曾吃掉 DiT 64% 算力）；N1/N2（坐标、路由）平局说明
> 「按内容自适应」必须内生于算子 → S6 的 Δ/B/C 由输入动态生成，就是原生选择机制。
>
> N3（对抗退化）仍在跑（明早出数）；本臂是**架构类**的正交赌注。

---

## 架构（MambaIR 式残差组，无金字塔）

```
x → stem 3×3 → [RG ×4] → LN → 3×3（零初始化） → + bicubic（训练器已有）
RG = Σ(VSSBlock ×4) → 3×3（零初始化）→ + identity
VSSBlock = LN → dwconv 3×3 → SiLU → SS2D → 1×1（零初始化）→ + identity
```

所有残差贡献者（块 proj / RG 尾卷积 / 输出头）零初始化 → **step 0 精确等于 bicubic 地板**，
与全仓库 reg 臂的不变量一致（smoke 已验证 trunk 恒等 + 输出恒 0）。

**SS2D 交叉扫描**：图像展平成 4 个扫描序（行主序 / 反向 / 列主序 / 反向），沿 batch 维拼接
一次前向，输出逆复原后均值融合 —— 消除一维扫描的方向偏见。

**参数量（实测，纠正预估）**：

| 配置 | 参数 | 说明 |
|---|---|---|
| c128 4×4（用户规格） | **2.75M** | 预估的 25–30M 有误：Mamba 块是投影型算子，块内 ≈134K |
| c192 4×4 | 5.99M | |
| c256 4×4 | **10.48M** | 对齐「容量拐点在 10M 以上」的实测；锚点 b64 是 18.67M |

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
| M1 `mamba_c128` | dim128 / 4RG×4 / N16 / E2，其余与锚点逐字相同 | 与锚点完全一致：**1.28M 样本**（CUDA 核：b128×10k；torch 回退：b64×20k） |
| M2 `mamba_c256` | 同上，dim256（10.48M） | b64×20k = 1.28M（仅 CUDA 后端时自动跑；torch 回退默认跳过，`RUN_C256=1` 强制） |

判定（纪律同 FFN 教训）：官方 100 对 + IQA 定生死，±0.05 dB 内按平局；
16 对 val 只看趋势。**M1/M2 任一裸跑破 34.11 即架构路线成立**；若 c256 明显高于 c128，
说明 mamba 类在 390 对上也吃容量，与 U-Net 同规律。

## 排队与状态

- 等 `newarms` 会话（N3 对抗预训练链，ETA 明早 ~05:10）结束后自动拉起（tmux `mambaarm`）。
- 日志：`experiments/diffusion/logs/mb_*.log`、`mamba_arm.log`；结果：`experiments/diffusion/mamba/<臂>/eval_iqa/eval.json`。

## 已知边界（明说）

- Mamba 是序列依赖的，官方评估仍走 tile 64/pad 16（cross-fade 32px 抑制接缝）——与其它臂同协议；
  若 M 臂出数接近锚点，需补一个 pad 32 的子集敏感性检查。
- `torch` 回退的纯扫描在 4090 上约 1–2 s/step（b64），M1 全程 ~10h；官方核成功则 ~1.5h。
- flow / latent 不支持（scan 忽略 t，reg-only，启动即校验拒绝）。
