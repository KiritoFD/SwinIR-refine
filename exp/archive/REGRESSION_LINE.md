# 回归线（SwinIR / ModSwinIR）：有效改动与指标增量

**协议：** RealSR V3 ×2 · 406 训练对 · 12 张 tiled Test（主表）· RTX 4070 8GB  
**基准锚点：** 官方 SwinIR-largeish 3.96M + 纯 L1 → **RGB 31.16**（12 张）

> 增量为同协议相对量；6k 消融里 ≤0.03 dB 视为噪声。全量 100 对与 12 张有 ~0.5 dB 级差，**只比同表**。

---

## 1. 有效改动一览（按贡献排序）

| 顺位 | 改动 | 相对谁 | **Δ PSNR (12张 RGB)** | 备注 |
|------|------|--------|------------------------|------|
| 1 | **Offset-aligned L1**（训练期 ±3px） | 纯 L1 + 同主干 | **+0.57**（31.16→31.73） | 唯一确认的结构性涨点 |
| 2 | **更长 schedule** 5k→12k（Align 配方） | 同模型 5k | **≈ +0.6～0.8**（6 张在线 31.5→32.3；12 张亦升） | 低数据下预算主导 |
| 3 | **L1 替代 UWCL/CX** | UWCL 主损失 | **+0.95**（30.14→31.09） | 损失选择，非结构 |
| 4 | **残差 bicubic 重建** | 绝对 RGB 预测 | 条件化（step0 PSNR **4.6→28.8**） | 必备；未做严格 A/B |
| 5 | **EMA 0.999** | 无 EMA @6k | **≈0**（短训早期还慢） | 长训/最终 ckpt 用 |
| 6 | **AMF tanh / FiLM-RAPE / OCA / GDFN / DCN 主干** | 官方同容量 SwinIR | **≈0～+0.1** | 31.09 vs 31.16（噪声级） |
| 7 | LP-KPN 叠在 Align 上 | E11 | **+0.02** | 中性 |
| 8 | Wiener / AmpPhase / RadialPSF | E11 | **−0.6～−1.0** | 负收益，已关 |

---

## 2. 关键数字链（可引用）

| 配置 | RGB PSNR | Y PSNR | 说明 |
|------|----------|--------|------|
| SwinIR-light 0.61M L1 | 31.07 | — | 官方结构 |
| SwinIR-largeish 3.96M L1 | **31.16** | — | 容量对照 |
| Mod v2 L1+EMA（无 Align） | 31.09 | — | 主干现代化 ≈ 打平 |
| **Mod v2 + Align（推荐）** | **31.73** | **32.40** | E11，12k |
| 全量 100 对（Align 配方） | ~31.7 RGB | — | E9 风格 |

**推荐 ckpt：** `experiments/improve/E11_align_loss/ckpt_best.pt`

---

## 3. 分项说明（为何是这些 Δ）

### 3.1 Align-L1（+0.57 dB）— 主贡献
RealSR 残余 1–3px 非刚性错位使像素 L1 比较错坐标。  
\[
\mathcal L=(1-\alpha)\|\hat y-W(y,\mathrm{off})\|_1+\alpha\|\hat y-y\|_1+\lambda\|\mathrm{off}\|_1
\]
推理图不变；**零部署成本**。

### 3.2 训练预算（+0.6～0.8 dB）
406 对下 4k→12k 仍明显上升；与文献 100k+ 的差距主要是预算，不是缺头。

### 3.3 损失：L1 ≫ UWCL（+0.95 dB）
CX 类在 RealSR 迁移可掉 2–3 dB；PSNR 主线用 L1（可选极轻 amp）。

### 3.4 残差 bicubic（条件化）
\(\hat y=\mathrm{bicubic}(x)+f_\theta(x)\) 把任务变成「学真实相机相对双线性的残差」，优化上远好于绝对 RGB。

### 3.5 主干现代化（≈0）
同容量下 OCA/GDFN/AMF/FiLM/DCN **没有**相对官方 SwinIR 拉开差距；说明该 regime **数据/目标 > 模块**。

---

## 4. 推荐生产配方

```
ModSwinIR-base 4.05M
  residual bicubic + Align-L1 + EMA 0.999
  batch=8, LR patch=64, 12k steps, AMP fp16
  eval: RGB + Y (官方 RealSR 风格)
```

```powershell
python -m model.train --data-root "G:\RealSR\data\RealSR(V3)" `
  --out experiments\E11 --scale 2 --model-size base `
  --batch-size 8 --lr-patch 64 --steps 12000 `
  --align-loss --ema 0.999 --amp --eval-every 1500
```

---

## 5. 尚未闭合的增量（勿当已验证）

| 项 | 预期 | 状态 |
|----|------|------|
| Muon vs AdamW | 不确定 | 已接入，无 RealSR A/B |
| 30k–50k step | +1 dB 量级？ | 未跑 |
| ×3 / ×4 | 同配方 | 未跑 |
| 全量 100 对稳定 + Align 对照 | — | E9 有 100 对；E11 仅 12 张 |

细节与失败项：`SWINIR_LINE.md`、`NEGATIVE.md`。
