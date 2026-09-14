# 回归线验证汇总（代码只含有效项）

**验证日：** 2026-09-14  
**代码：** `model/` — 仅 L1 / Align / EMA / 残差 bicubic / Muon；**无** Wiener·AmpPhase·RadialPSF·UWCL  
**协议：** RealSR V3 ×2，12 张 tiled Test

---

## 1. 验证结果（本轮复跑）

| 检查 | 结果 |
|------|------|
| `build_model` + Align 前向反向 | PASS（4.05M） |
| `model.train` 4-step smoke | PASS |
| `model.eval` 加载 E11 | PASS（忽略旧 ckpt 里 kpn/unc_head 多余键） |
| E11 12 张 | **RGB 31.73 / SSIM 0.902 · Y 32.40 / 0.910** |

---

## 2. 各项「提升多少」（不可线性相加）

| 组件 | 相对基线 | ΔPSNR (12张 RGB) | 可加性 |
|------|----------|------------------|--------|
| 官方 SwinIR-largeish + L1 | — | **31.16**（基线） | — |
| 主干 OCA/GDFN/AMF/FiLM/DCN | vs 官方同容量 | **≈0**（31.09 vs 31.16） | 噪声级 |
| L1 替代 UWCL | vs UWCL | **+0.95** | 独立于 Align |
| EMA | vs 无 EMA @6k | **≈0** | 噪声级 |
| **Offset-aligned L1** | vs 同主干纯 L1 | **+0.57** | 主增量 |
| 更长 schedule 5k→12k | vs 同配方 5k | **≈+0.6～0.8** | 与 Align 叠加 |

**不要把表内 Δ 直接相加**（非正交、有交互）。已验证的**合成结果**是单一配方 E11。

---

## 3. 最优模型总提升（已验证）

| 指标 | 官方 SwinIR-largeish L1 | **最优 E11** | **总提升** |
|------|-------------------------|--------------|------------|
| RGB PSNR | 31.16 | **31.73** | **+0.57** |
| Y PSNR | （未单列） | **32.40** | — |
| RGB SSIM | 0.890 | **0.902** | +0.012 |

另一口径：相对 **UWCL 版 Mod**（30.14 RGB）→ E11 **+1.59 dB**（损失+对齐+预算混在一起）。

**最优配方：** 残差 bicubic + Align-L1 + EMA + 12k steps  
**ckpt：** `experiments/improve/E11_align_loss/ckpt_best.pt`

---

## 4. 代码 vs 文档边界

| 位置 | 内容 |
|------|------|
| `model/` | 仅有效路径（Align 默认开，可 `--no-align-loss`） |
| `exp/NEGATIVE.md` | UWCL、Wiener、AmpPhase、RadialPSF、错误 AMP 等 |
| `exp/REGRESSION_LINE.md` | 分项增量与实验史 |
