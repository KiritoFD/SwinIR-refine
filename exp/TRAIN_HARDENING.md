# 2026-09-15 代码加固（Muon 可跑通）

## 问题

1. **dataset 全图 RGB 转换**：`RealSRPairDataset` 先对整幅 HR（可达 2k+）`convert("RGB")` 再 crop，Windows 上易推高 RAM，和 GPU 任务抢资源。
2. **无梯度累积**：8GB 卡上 Muon + Align 训练想对齐 E11 的 effective batch=8，需要 `batch=4 × grad_accum=2`。
3. **日志未 detach**：`float(total)` 在带 grad 的 tensor 上触发 UserWarning。
4. **E11c_muon OOM**：训练内 eval tile=128 在大图上峰值过高；`save-every=4000` 且 best 只在 eval 成功时写，OOM 后 **没有可用 ckpt**。

## 改动

| 文件 | 改动 |
|------|------|
| `model/dataset.py` | 只对 crop 后的 patch `convert("RGB")`，避免全图转换 |
| `model/train.py` | `--grad-accum`；`total.detach()` 打日志 |
| `scripts/resume_muon.ps1` | 低显存 Muon 启动：tile=96, pairs=4, 无 AMP |
| `scripts/check_ckpts.py` | 用当前 slim 包批量 load-test 全部实验 ckpt |

## ckpt 兼容性（`python scripts/check_ckpts.py`）

**可加载（16）**：A0–A4, E7, E8, E10, E11, E12–E15, E1/E2/E5 SwinIR

**不可加载（4，已记入 OFFICIAL_EVAL.md）**：

- E6 / E6b / `mod_swinir_x2_base`：旧 RAPE 形状 `rape.mlp.2` 96 vs 192
- E4_mod_large：`model_size=large` 已从 slim 包移除

## Muon A/B 配方（对齐 E11）

| 项 | E11 (AdamW) | E11c (Muon) |
|----|-------------|-------------|
| 主干 | Mod base 4.05M | 同 |
| loss | Align-L1 + EMA 0.999 | 同 |
| steps / patch | 12k / 64 | 同 |
| batch | 8 | 4 × accum 2 = 8 |
| AMP | fp16 on | **off**（Muon NS 在 bf16，与 GradScaler 叠易 CUDA unknown） |
| lr | 2e-4 | 1e-3（Muon 常用） |
| seed | 42 | 42 |

## 协议提醒

官方全量：`python -m model.eval --max-pairs 0` → RealSR V3 Test.m，Y limited-range uint8，crop=0，modcrop=4。见 `exp/EVAL_PROTOCOL.md`。
