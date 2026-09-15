# 官方评估进度（2026-09-15 续）

**约束：** 串行、单模型、tile=96、SR 缓冲放 CPU，目标显存峰值 **&lt;4GB**  
**协议：** RealSR V3 Test.m（Y limited-range uint8, crop=0, modcrop=4）  
**批目录：** `experiments/eval_official/batch_20260915/`  
**驱动：** `scripts/run_official_evals_serial.py`（每模型独立子进程，可断点续跑）

## 已完成（batch_20260914，权威）

| 模型 | n | Y-PSNR | Y-SSIM | RGB |
|------|---|--------|--------|-----|
| E11 Align-L1 EMA | 100 | **33.47** | 0.9144 | 31.62 |
| A0 pure L1 | 100 | 33.37 | 0.9124 | 31.51 |
| E2 SwinIR-largeish | 100 | 32.97 | 0.9058 | 31.13 |
| E9 Mod L1 long 12k | 100 | 32.92 | 0.9091 | 31.11 |
| E1 SwinIR-light | 100 | 32.88 | 0.9026 | 31.06 |
| E5 SwinIR-classical | 100 | 31.51 | 0.8784 | 29.46 |

## 本批补齐中（batch_20260915）

目标：E10, A1, A2, A3, A4, E7, E8, E12, E13, E14, E15 → 全量 100 对

| 模型 | 状态 |
|------|------|
| E10_realsr_loss | 跑批中（子集 n=30 曾报 33.86，待全量） |
| 其余 | 排队 |

## 不可加载（文档保留，不跑）

| ckpt | 原因 |
|------|------|
| E6 / E6b / mod_base_early | 旧 RAPE `rape.mlp.2` 96 vs 192 |
| E4_mod_large | `model_size=large` 已移除 |
| E11b_muon | 训练未完成，无可用 best |

## 上一轮失败原因（已修）

1. tile=192 + 整幅 SR 放 GPU → RAPE FiLM 峰值过高
2. `psnr_uint8` float64 大图 → 系统 RAM OOM（2296×2496×3）
3. 多进程并行评估 → 显存叠爆
4. 页面文件压力下 torch DLL 加载失败（WinError 1455）

修复：CPU 累加 SR、metrics float32、串行单进程、detach 启动。

## 命令

```powershell
# 串行全量评估（可续跑）
python scripts\run_official_evals_serial.py

# 单模型
python -m model.eval --ckpt <path> --max-pairs 0 --out <dir> --tile 96
```
