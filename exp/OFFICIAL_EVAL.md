# 官方协议评估结果（2026-09-14）

**协议：** RealSR V3 Test.m（limited-range Y, uint8, crop=0, modcrop=4, Canon+Nikon）  
**代码：** `model/eval.py` + `model/metrics.py`  
**批目录：** `experiments/eval_official/batch_20260914/`

## 全量 100 对（可横比）

| 模型 | 配置 | **Y-PSNR** | Y-SSIM | RGB-PSNR |
|------|------|------------|--------|----------|
| **E11 Align-L1 EMA** | Mod base 4.05M, 12k, Align | **33.47** | **0.9144** | 31.62 |
| A0 pure L1 | 同主干无 Align, 6k | 33.37 | 0.9124 | 31.51 |
| E2 SwinIR-largeish L1 | ~10M 官方结构, 4k | 32.97 | 0.9058 | 31.13 |
| E9 Mod L1 long | 无 Align, 12k | 32.92 | 0.9091 | 31.11 |
| E1 SwinIR-light | ~1.2M, 4k | 32.88 | 0.9026 | 31.06 |
| E5 SwinIR-classical | ~12M, 4k | 31.51 | 0.8784 | 29.46 |

### 官方全量结论

- **E11 最优：** Y **33.47** / SSIM **0.9144**
- **Align 增量（同主干、可比）：** 33.47 − 33.37 = **+0.10 dB Y**（比 12 张 RGB 子集的 +0.57 小；子集会放大噪声）
- vs SwinIR-capmatch：**+0.50 dB Y**
- vs SwinIR-light：+0.59 dB Y
- 旧 12 张 full-range Y 32.40 **不可**与本表横比

## 子集（仅内部参考，n 不同勿横比全量）

| 模型 | n | Y-PSNR |
|------|---|--------|
| E10 RealSR loss | 30 | 33.86* |
| A1 L1+EMA | 30 | 33.63 |
| E12 LP-KPN | 20 | 33.13 |
| A4 full | 30 | 33.22 |
| E13 Wiener | 20 | 32.51 |
| E14 AmpPhase | 20 | 31.79 |
| E15 RadialPSF | 20 | 31.78 |

\* n=30 子集偏易；**不能**据此宣布 E10>E11。E13–E15 仍差于 Align 主线。

## 未纳入

| ckpt | 原因 |
|------|------|
| E6 / mod_base_early | 旧 RAPE 形状与当前 `model/` 不兼容 |
| E4_mod_large | `model_size=large` 已从 slim 包移除 |

## Muon A/B

- 训练中：`experiments/improve/E11c_muon`（AdamW→Muon, lr 1e-3, Align+EMA 12k, 无 AMP）
- 完成后用同协议 `model.eval --max-pairs 0` 对比 E11 的 **Y 33.47**
