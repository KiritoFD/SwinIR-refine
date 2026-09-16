# 全量消融评估协议（改进项）

**日期：** 2026-09-15  
**协议：** RealSR V3 Test.m 官方全量 100 对（Y limited-range uint8, crop=0, modcrop=4）  
**代码：** `model/eval.py`（tile=96, `--resume` 分图断点）  
**矩阵：** `scripts/ablation_matrix.json`

## 我们在消融什么

相对生产配方 **E11**（Mod base + Align-L1 + EMA 0.999 + 12k + AdamW）：

| 因子 | 对照 | 已有 ckpt | 全量结果 |
|------|------|-----------|----------|
| **Align-L1** | E9 关 Align，其余同 E11 | 有 | E11 33.47 vs E9 32.92 → **+0.55** |
| **EMA @6k** | A0 无 EMA vs A1 有 EMA | 有 | 33.37 vs 33.38 → **≈0** |
| **Muon** | E11c 同配方换优化器 | **待训** | — |
| **LP-KPN 头** | E12 = E11 + LP-KPN | 有 | 33.00 → 弱于 E11 |
| **Wiener/AmpPhase/Radial** | E13–E15 | 有 | 31.7–32.4 → 负收益 |
| **patch-amp / HF-conf** | A2 / A3 vs A1 | 有 | ≈0 |
| **结构对照** | SwinIR-light / capmatch | 有 | E11 **+0.49** vs capmatch |

## 命令

```powershell
# 1) 串行全量评估矩阵（可续跑；已 n>=100 会跳过）
python scripts\run_ablation_evals.py

# 2) 只补缺的
python scripts\run_ablation_evals.py --only E7_v2_l1_amp_ema,E8_v2_kpn,E11c_muon

# 3) 汇总所有 batch → 总表
python scripts\aggregate_full_results.py
```

## Muon A/B 训练（对齐 E11）

```powershell
# 无 AMP，batch4×accum2=8，lr 1e-3，eval tile 96
powershell -File scripts\run_muon_stable.ps1
# 或
scripts\run_muon_stable.bat
# 后台 detach
python scripts\detach_run.py G:\RealSR\scripts\run_muon_stable.bat G:\RealSR\experiments\improve\E11c_muon
```

训完后再跑 `run_ablation_evals.py --only E11c_muon` 与 `aggregate_full_results.py`。

## 输出位置

| 文件 | 内容 |
|------|------|
| `experiments/eval_official/ablation_full_20260915/<id>/eval.json` | 单模型全量 |
| `experiments/eval_official/ablation_full_20260915/summary.json` | 矩阵评估状态 |
| `experiments/eval_official/ablation_full_20260915/RESULTS.md` | 表 + Δ vs E11 |
| `experiments/eval_official/OFFICIAL_FULL_SUMMARY.md` | 跨 batch 总表 |

## 显存约束

- 单进程串行，tile=96，SR 缓冲在 CPU
- metrics float32
- 实测峰值约 **1.1GB**（远低于 4GB 预算）
- 不要并行多个 `model.eval`
