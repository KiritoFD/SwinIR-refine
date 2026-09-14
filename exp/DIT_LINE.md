# DiT 线：现代化起点（占位）

**目标：** 在 RealSR V3 上做 **确定性 / 少步** DiT 风格 SR（非全量扩散采样），与 SwinIR 线并行。

## 为何开这条线

- 文献与 sr-scaling：pixel diffusion **PSNR 弱于** 回归 SwinIR，但 DiT 结构（patchify + AdaLN / 条件注入）在别的域很强。  
- 合理定位：**回归式 DiT / 一致性蒸馏**，而不是完整 DDPM 采样拼 PSNR。

## 设计约束（与 SwinIR 线对齐）

| 项 | 约定 |
|----|------|
| 数据 | 同一 RealSR V3 ×2，406 对 |
| 目标 | Y-PSNR 主线；可选 LPIPS 旁路 |
| 训练 | 残差 bicubic + **Align-L1**（已验证）可作为 DiT 目标 |
| 显存 | 8GB：小 patch DiT，~4–10M 参数 |

## 候选（待实现）

1. **DiT-SR 回归版**：LR patchify → Transformer blocks（AdaLN-zero 条件 scale）→ unpatchify 残差。  
2. 对照：同参数量 ModSwinIR-base（已 31.73/32.40）。  
3. 明确不做：多步采样 diffusion 作为 PSNR 主线。

## sr-scaling 包内 DiT 审计（2026-09-14）

代码：`sisr_scaling/arch_diffusion.py` `_DiTDenoiser` / `PixelDiffusionDiT`  
包内文档（`AGENTS.md` / `docs/current-status.md`）自认 **`dit_lite`**。

| 组件 | 原版 DiT / 现代 SR-DiT | 包内实现 |
|------|------------------------|----------|
| Patchify | ✓ | ✓ `Conv2d(patch,patch)` + `ConvTranspose2d` unpatch |
| Transformer block | Pre-LN + AdaLN-Zero | `nn.TransformerEncoderLayer` **norm_first**，无 AdaLN |
| 时间条件 | AdaLN 调制 scale/shift/gate | **加性 token bias** `tokens + time_emb` |
| 位置编码 | RoPE / sincos 2D PE | **无显式 PE** |
| 训练目标 | flow-matching / v-pred / EDM 等 | **DDPM ε / x0**，或 ResShift 4-step residual shift |
| 采样 | Euler / Heun flow, 少步 | DDIM；ResShift 路径 `steps≈4` |
| 分辨率/扩展 | adaLN + RoPE 可外推 | 固定 patch grid，无 RoPE |

**评估：** 作为对照 denoiser 家族可以留；**不能**当 2024–2026 现代 DiT/flow 基线。若开 DiT 线，应自建：
1. AdaLN-Zero 条件（t 或 LR 特征）
2. 2D RoPE 或 sincos PE
3. 可选 rectified-flow / 少步确定性目标（PSNR 主线仍建议回归 + Align）

## 状态

- [ ] Spec / 最小 DiT 回归模型（AdaLN + RoPE，非 dit_lite）  
- [ ] 与 Align 配方对齐的 train 入口  
- [ ] 6k–12k 步 A/B  

详细实现见后续 spec。
