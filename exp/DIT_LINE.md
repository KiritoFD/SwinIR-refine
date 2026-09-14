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

## 状态

- [ ] Spec / 最小 DiT 回归模型  
- [ ] 与 Align 配方对齐的 train 入口  
- [ ] 6k–12k 步 A/B  

详细实现见后续 spec。
