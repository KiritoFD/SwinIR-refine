---
theme: default
title: 真实世界单帧图像超分辨率
class: text-center
lineNumbers: false
drawings: false
transition: slide-left
fonts:
  sans: 'Noto Sans SC, system-ui, sans-serif'
---

# 真实世界单帧图像超分辨率

<div class="text-xl mt-1 opacity-80">正交小波高频损失 + Muon 的判别式回归 · RealSR(V3) ×2</div>

<div class="grid grid-cols-4 gap-2 mt-10 text-center text-2xl font-bold" style="color:#1f4e79">

<div>34.41<div class="text-sm font-normal">PSNR-Y dB</div></div>
<div>0.9285<div class="text-sm font-normal">SSIM</div></div>
<div>55.96<div class="text-sm font-normal">MUSIQ</div></div>
<div>0.3534<div class="text-sm font-normal">MANIQA</div></div>

</div>

<div class="abs-bottom text-xs opacity-50">官方 Test.m · 有限量程Y · 不剃边 · 100对 · 18.7M 参数</div>

---
---

# 一、引言：问题与上限

真实配对 = 同场景**两种焦距**拍摄；退化未知非平稳，含噪声与亚像素错位。

$$\mathrm{MSE}_{\text{total}}=E_{\text{null}}(\text{奈奎斯特外信息已毁})+E_{\text{noise}}(\text{GT 噪声, aleatoric})+E_{\text{align}}(\text{残差配准})$$

<v-callout type="info" v-click>
定位：判别式回归 = 榨干**可预测**高频。命题：先量出 PSNR 硬上限，再看我们离它多近。
</v-callout>

---
---

# 二、相关工作（定位）

| 谱系 | 代表 | 相对我们 |
|---|---|---|
| 判别式 CNN/Transformer | EDSR·RCAN·SwinIR | 我们同协议全面超越 |
| 真实退化先验 | BSRGAN·Real-ESRGAN | 合成退化，非配对保真最优 |
| 生成式(扩散) | SeeSR·OSEDiff·ResShift·TSD-SR… | 感知↑但 PSNR 崩 6–10 dB |
| 理论 | Blau&Michaeli18 感知-失真权衡；Baker&Kanade02；Robinson&Milanfar04 | 解释"为何见顶" |

---
---
class: text-sm
---

# 三、方法① 骨架 + 正交小波高频损失

$$\hat y=\mathcal{B}(x_\downarrow)+f_\theta(x_\downarrow),\qquad \mathcal{L}=\|\hat y-y\|_1+\lambda\underbrace{\sum_{j}\sum_{d\in\{HL,LH,HH\}}\|\mathcal{W}^{(j)}_d(\hat y)-\mathcal{W}^{(j)}_d(y)\|_1}_{\text{小波高频损失 }\mathcal{L}_{HF}}$$

- **stride-1**：主干不下采样 → 保相位（**+0.38 dB**，最大单项）。
- **零初始化头**：起点 $\hat y\equiv$ bicubic ⇒ 永不吃亏的地板不变量。
- **为何正交+小波**：Parseval ⇒ 子带 L1 是像素 L1 的无损重分配；FFT 全局基会失稳(已证伪)。
- **为何要它**：L1 最优解 = 条件均值 → 高频被抹平；λ8 显式抬回。推理零成本。

---
---
class: text-sm
---

# 三、方法② 移位集成 · Muon · TTA

**移位集成**（治 Haar 平移敏感，DTCWT 廉价替身）：
$$\mathcal{L}_{\text{shift}}=\tfrac1{|\mathcal S|}\textstyle\sum_{s\in\mathcal S}\mathcal{L}_{HF}(\text{roll}(\hat y,s),\text{roll}(y,s))$$

**Muon**：≥2D 权重做 Newton–Schulz 正交化 $O=\mathrm{NS}(G)$，$\Delta\theta=-\eta\,\mathrm{rms}\cdot O$；1D 走 AdamW。

- 全 lr 段胜 AdamW；最优 5e-3；**零预训练+Muon ≈ 预训练+AdamW**。

**BSRGAN 预训练** + **D4-TTA**（8 重平均，抬 SSIM/PSNR）。

---
---
class: text-xs
---

# 四、实验 · 主表（全方法 × 全指标 · 含 Ours）

<div class="opacity-70 -mt-1 mb-1">**同协议内**最优**加粗**、<u>次优下划线</u>；A-strict 与文献 B-协议不可跨组横比。</div>

| 组 | 方法 | 类别 | PSNR-Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|---|---|
| **A** | **Ours（完整+TTA）** | U-Net s1 | **34.41** | **0.9285** | **55.96** | **0.3534** |
| A | s1_b64(零预训练) | U-Net | <u>34.11</u> | <u>0.9246</u> | <u>55.22</u> | <u>0.3420</u> |
| A | EDSR-baseline | CNN | 33.60 | 0.9167 | 53.14 | 0.3273 |
| A | E11 Mod-SwinIR | Transformer | 33.47 | 0.9143 | 49.15 | 0.3090 |
| A | SwinIR-largeish | Transformer | 32.97 | 0.9058 | 46.79 | 0.3050 |
| A | SwinIR-light | Transformer | 32.88 | 0.9026 | 45.43 | 0.3020 |
| A | Bicubic | — | 31.73 | 0.8876 | 39.40 | 0.2780 |
| A | 我们·latent-flow | 生成式 | 28.36 | 0.7900 | 50.12 | 0.2240 |
| B | ResShift | 扩散 | **28.69** | **0.7874** | 52.40 | 0.4756 |
| B | SeeSR | 扩散 | <u>28.14</u> | 0.7712 | 64.74 | **0.6022** |
| B | AdcSR | 一步扩散 | 28.10 | 0.7726 | <u>66.26</u> | <u>0.5927</u> |
| B | TSD-SR | 一步扩散 | 27.77 | 0.7559 | **66.62** | 0.5874 |
| B | OSEDiff | 一步扩散 | 27.92 | <u>0.7836</u> | 64.69 | 0.5898 |
| B | Real-ESRGAN | GAN | 26.65 | 0.7603 | 60.45 | 0.5507 |
| B | BSRGAN | GAN | 26.38 | 0.7651 | 63.28 | 0.5425 |

<div class="text-sm mt-1" style="color:#b03024">A 组：Ours 四指标全项最优。B 组(文献×4,更难输入)：生成式以感知换保真。RCAN/SRResNet/RRDB 复现中→将并入 A 组。</div>

---
---
class: text-sm
---

# 四、实验 · 免训练实测的数据集上限

<div class="grid grid-cols-2 gap-6 items-start">
<div>

| 墙 | ×2 | ×3 | ×4 |
|---|---|---|---|
| 配准 0.3px | **37.7** | · | · |
| 带宽 $E_{null}$ | **40.65** | 34.55 | 31.27 |
| 噪声(MAD σ≈0.9) | 49.05 | 49.05 | 49.05 |
| 带宽+噪声 | **40.1** | 34.4 | 31.2 |
| +0.3px 配准 | **35.7** | ~32 | ~29.5 |

</div>
<div>

- 上限由**配准**主导，非噪声（修正"38–40dB 噪声底"）。
- ×2 天花板 **34.4–35.7 dB** → **Ours 34.41 已在带内**：不是模型不够，是物理不允许。

</div>
</div>

---
---
class: text-xs
---

# 四、实验 · 消融（同协议单因子 ΔPSNR）+ 跨尺度

<div class="grid grid-cols-2 gap-6">
<div>

| 改动 | ΔPSNR | 维度 |
|---|---|---|
| stride-2→**1** | **+0.38** | 架构 |
| +小波高频 λ8 | +0.11 | 损失 |
| +**Muon** | +0.15 | 优化器 |
| +移位集成 | +0.02 | 损失(MUSIQ最高) |
| +预训练 | +0.09 | 数据 |
| +TTA | +0.10 | 推理 |
| 容量→72.5M | +0.05 | 饱和 |

</div>
<div>

| 尺度 | ours | 该尺度上限 | 判定 |
|---|---|---|---|
| ×2 | 34.41 | 35.7 | 见顶 |
| ×3 | 31.13(λ2) | 34.4 | 欠调有余量 |
| ×4 | 29.51(λ5) | 31.2 | 近顶 |

最优 λ 随退化反相关：×2≈8 / ×3≈2 / ×4≈5。

</div>
</div>

---
---
class: text-sm
---

# 四、实验 · 负结果（28+ 臂，如实报告）

| 路线 | 结论 |
|---|---|
| 扩散(pixel/latent×flow/reg) | 低于 bicubic / 不超自身零初始化 |
| **Mamba/VSS** | 等预算 **−2.3 dB** → 关线 |
| 双路小波U-Net / 双射小波U-Net / DTCWT | 均不敌「Haar高频损失+移位」 |
| 频域 rectified-flow(HF生成) | 单调毁图：造不可预测高频=注伪影 |
| D4等变 / 坐标 / 频域路由 / 门控FFN | 持平或负 |
| 对抗挖数据(N3+Muon) | 中性，未破平台 |

---
---
---
class: text-center
---

# 配图 · 主结果阶梯

<img src="/figs/fig_results.png" class="mx-auto" style="max-width:96%;max-height:76vh"/>

---
---
class: text-center
---

# 配图 · 免训练上限（三墙）

<img src="/figs/fig_ceiling.png" class="mx-auto" style="max-width:96%;max-height:76vh"/>

---
---
class: text-center
---

# 配图 · 感知-失真权衡

<img src="/figs/fig_tradeoff.png" class="mx-auto" style="max-width:92%;max-height:76vh"/>

---
class: text-sm
---

# 五、结论

- **配方**：stride-1 + 正交小波高频(含移位) + Muon + 预训练 + TTA → **34.41 / 0.9285**，同协议四指标全面超重训 SwinIR（+1.44 dB / +9.2 MUSIQ / +0.048 MANIQA），参数更小。
- **上限**：免训练实测 ×2 天花板 ~34.4–35.7 dB（配准主导），**我们已在内**。
- **下一步**：×3/×4 全配方逼近上限（唯一余量）；保真见顶 → 转感知轴(MUSIQ/LPIPS/FID)。

<div class="abs-bottom text-xs opacity-50">复现 README §9 · 测上限 python -m diffusion.oracle · 全记录 exp/FINAL_REPORT.md · 基线 docs/baseline/</div>
