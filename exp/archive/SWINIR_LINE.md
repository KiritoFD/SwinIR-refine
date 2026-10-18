# SwinIR 线：已确认有效的改进（整理）

**日期：** 2026-09-14 · **任务：** RealSR V3 ×2 · **硬件：** RTX 4070 8GB · **数据：** 406 训练对

---

## 1. 一句话结论

在官方 SwinIR 思路上，**真正涨点的是「训练目标对准真实退化」**：残差 bicubic 重建 + **训练期 offset-aligned L1**；主干现代化（OCA/GDFN/AMF/FiLM/DCN）与同容量 SwinIR 打平或略优，**不是**再堆频域头。

---

## 2. 相对官方 SwinIR 的有效改动

| # | 改动 | 数学/物理动机 | 证据 |
|---|------|----------------|------|
| 1 | **残差 bicubic 重建** \(\hat y=\mathrm{bicubic}(x)+f_\theta(x)\)，头零初始化 | 只学真实相机相对双线性的残差；step0 PSNR 4.6→28.8 | 所有后续 run |
| 2 | **Offset-aligned L1**（±3px warp HR→SR + 纯 L1 锚） | RealSR 残余错位让像素 L1 比错坐标；对齐后再罚 | E11 vs E9：**+0.57 dB RGB** |
| 3 | **OCA + GDFN + AMF(tanh) + FiLM-RAPE + DCN** | 空间可变 PSF、窗边界、局部噪声 | 与同容量 SwinIR ≈ 持平略优 |
| 4 | **EMA 0.999 + L1 主损失** | 稳定收敛；UWCL/CX 有害 | E6 vs 旧 UWCL |
| 5 | **Y 通道评估** | 官方 RealSR：RGB 训练 / Y 测试 | 与文献对齐，差 ~0.6–0.7 dB |
| 6 | **Muon（可选）** | sr-scaling 主线优化器 | 已接入 `--optimizer muon`，未在 RealSR 上完成 A/B |

### 数字（12 张 tiled Test，RGB / Y）

| 配置 | RGB | Y |
|------|-----|---|
| 官方 SwinIR-light 0.61M L1 | 31.07 | — |
| 官方 SwinIR-largeish 3.96M L1 | 31.16 | — |
| Mod v2 + Align **（推荐）** | **31.73** | **32.40** |
| 全量 100 对 RGB（E9 风格） | ~31.7 | — |

**权重：** `experiments/improve/E11_align_loss/ckpt_best.pt`

---

## 3. 已否定（不要再默认打开）

| 做法 | 结果 |
|------|------|
| UWCL / CX 主损失 | **−0.95 dB** |
| Wiener / AmpPhase / RadialPSF 残差头 | 均 **差于** 仅 Align |
| LP-KPN 叠在 Align 上 | 中性（≈E11） |
| 10M + fp16 + 2e-4 | NaN |
| batch=1 却不加倍步数 | 欠训练 |

细节：`NEGATIVE.md`。

---

## 4. 生产配方（当前默认）

```powershell
python -m model.train --data-root "G:\RealSR\data\RealSR(V3)" `
  --out experiments\E11 --scale 2 --model-size base `
  --batch-size 8 --lr-patch 64 --steps 12000 `
  --align-loss --ema 0.999 --amp --eval-every 1500
```

容量甜点：**4–12M**；再涨点优先 **更长 schedule / 更大有效 batch**，不是新头。

---

## 5. SwinIR 线下一步（可选）

1. 30k–50k step 复跑 Align 配方，逼近文献 33+ Y（需更多 GPU 时）。  
2. Muon vs AdamW 在 RealSR 上的正式 A/B。  
3. ×3 / ×4 同配方。  
4. 大 patch（官方 192 HR）若显存允许。
