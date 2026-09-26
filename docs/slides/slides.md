---
theme: default
title: 真实世界单帧图像超分辨率
class: text-center
lineNumbers: false
drawings: false
transition: slide-left
fonts:
  sans: 'Noto Sans SC, system-ui, sans-serif'
  mono: 'JetBrains Mono, Consolas, monospace'
---

# 真实世界单帧图像超分辨率

### 正交小波高频损失 + Muon 优化的判别式回归方法

<div class="mt-6 text-lg">
数据集 RealSR (V3) · 倍率 ×2 · 官方 <code>Test.m</code>（有限量程 Y / 不剃边 / 100 对）
</div>

<v-hoc class="gap-2 flex justify-center mt-6 text-2xl font-bold" style="color:#1f4e79">

<div>PSNR-Y 34.41 dB</div><div>·</div><div>SSIM 0.9285</div><div>·</div><div>MUSIQ 55.96</div><div>·</div><div>MANIQA 0.3534</div>

</v-hoc>

<div class="abs-bottom text-sm opacity-60">参数量 18.7M · Slidev</div>

---
layout: default
class: text-center
---

# 大纲

<div class="grid grid-cols-2 gap-6 mt-8 text-left text-lg leading-relaxed">

<div>

**一、引言**
- 真实超分的问题本质
- 三大不可约误差 → PSNR 硬上限

**二、相关工作**
- 判别式 / GAN / 扩散 / 理论

</div>

<div>

**三、方法**（从数学与问题出发）
- stride-1 回归 · 正交小波高频损失
- 移位集成 · Muon · 预训练 · TTA

**四、实验**（高密度）
- 全方法×全指标大表 · 消融 · 上限实测 · 负结果

</div>

</div>

---
layout: default
---

# 一、引言：问题的本质

真实配对来自**同一台相机、两种焦距**（Canon 5D3 / Nikon D810）：短焦得低清 $x_\downarrow$，长焦（配准后）得高清 $y$。

与合成分辨率（bicubic 退化）的根本区别：退化**未知、非平稳**，含真实镜头模糊、传感器噪声与亚像素错位。

<v-callout type="info">
核心命题：这份数据集在像素保真度（PSNR）上的**天花板在哪里**？我们离它多远？——本报告用**免训练探针**量出这个上限，并证明本方法已逼近它。
</v-callout>

我们的定位是**判别式回归**（最大化保真度），而非生成式"造出合理高频"；这一点将由实验与理论双重佐证。

---
layout: default
---

# 一、引言：误差分解 = 理论上限的三大不可约项

$$\mathrm{MSE}_{\text{total}} = \underbrace{E_{\text{null}}}_{\text{信息零空间}} + \underbrace{E_{\text{noise}}}_{\text{传感器噪声}} + \underbrace{E_{\text{align}}}_{\text{亚像素配准残差}}$$

| 项 | 来源 | 结论 / 文献 |
|---|---|---|
| $E_{\text{null}}$ | 奈奎斯特：高于 LR 采样率的高频被物理摧毁，只能"猜" | Baker & Kanade, TPAMI 2002 |
| $E_{\text{noise}}$ | GT 自带散粒/暗电流噪声（aleatoric），完美模型也预测不了那一瞬噪声 | Chatterjee & Milanfar, TIP 2010（CRLB）|
| $E_{\text{align}}$ | 变焦双镜头透视/畸变不同，标定后仍余 $0.2\text{–}0.5$ px 非刚性错位 | Robinson & Milanfar, TIP 2004 |

$$\Rightarrow \text{真实 SR 的 PSNR 上限是有限值；}\ \text{本数据上由 } E_{\text{align}} \text{ 主导（§四实测证实）。}$$

---
layout: default
---

# 二、相关工作

| 谱系 | 代表 | 特点 |
|---|---|---|
| 判别式 CNN | VDSR / **EDSR** / **RCAN** / RDN | 追 PSNR，合成退化，真实场景偏糊 |
| 判别式 Transformer | **SwinIR** / HAT | 全局注意力，重训后仍是保真主力 |
| 真实退化先验 | **BSRGAN** / Real-ESRGAN | 二阶合成退化，泛化强但非配对保真最优 |
| 生成式（GAN/扩散） | StableSR · DiffBIR · SeeSR · ResShift · OSEDiff · SinSR · AdcSR · TSD-SR · PURE · RealSR-R1 | 感知强，但 PSNR 常暴跌 6–10 dB |
| 理论 | 感知-失真权衡 Blau & Michaeli CVPR18 · SR/配准 CRLB Robinson&Milanfar 04/06 · 去噪 CRLB Chatterjee&Milanfar10 · SR 极限 Baker&Kanade02 | 解释"为何见顶" |

---
layout: default
---

# 二、相关工作：本工作的定位

- **不**引入外部大数据、**不**用扩散——只在 RealSR 域内做**机制**创新。
- 力量花在数学上讲得通处：**正交**小波高频损失（非全局 FFT 增益）、**stride-1** 相位保真、**正交化**的 Muon 优化。
- 给出**可证伪**的经验上限：免训练探针量出三堵墙，证明 34.41 dB 已在天花板内。
- 与扩散 SOTA **不横比 PSNR**（协议/输入不同），仅定性对照 + 我们自有生成臂做同协议权衡自证。

<div class="text-sm opacity-70 mt-4">对比坐标：同协议、判别式——我们做到保真前沿，且参数远小于同代 Transformer。</div>

---
layout: default
---

# 三、方法 · 总体形式

以「双三次上采 + 残差学习」为骨架，回归式建模：

$$\hat y = \mathcal{B}(x_\downarrow) + f_\theta(x_\downarrow), \qquad \mathcal{L}(\theta)=\|\hat y-y\|_1+\lambda\,\mathcal{L}_{\mathrm{HF}}(\hat y,y)$$

**关键设计 1 — 零初始化输出头**：末层权重置 $0$，故训练起点 $\hat y\equiv\mathcal{B}(x_\downarrow)$，即 **bicubic 地板不变量**：任何劣化都会被早停挡在地板之上。

**关键设计 2 — stride-1**：主干全程在 HR 尺度运算，不做 lossy 下采样 ⇒ 保住高频**相位**。（实测本工作最大单项增益 +0.38 dB。）

评价协议：训练 RGB、测试有限量程 Y（BT.601），uint8，`modcrop4`，不剃边，100 对全量。

---
layout: default
---

# 三、方法 · 正交小波高频损失（核心创新）

**动机（数学）**：L1 回归的最优解是条件均值
$$\hat y^\star=\arg\min \mathbb{E}\|y-\hat y\|_1=\mathrm{median}(y\mid x)\ \xrightarrow{\text{高频}}\ \text{过平滑}$$
故需**显式抬升高频子带**。对 HR 与 $\hat y$ 各做 $J$ 级正交 Haar DWT，仅惩罚高频子带：

$$\mathcal{L}_{\mathrm{HF}}=\sum_{j=1}^{J}\ \sum_{d\in\{HL,LH,HH\}}\bigl\|\,\mathcal{W}^{(j)}_{d}(\hat y)-\mathcal{W}^{(j)}_{d}(y)\,\bigr\|_1$$

- **为何正交**：Parseval 保能量 ⇒ 子带 L1 是像素 L1 的**无损重分配**，只调频率侧重、不注入伪能量（FFT 全局基不满足，Wiener/AmpPhase/RadialPSF 全部证伪）。
- **为何局部**：小波紧支撑，可表达镜头非平稳退化。
- **推理零成本**：只改损失，不改网络。

---
layout: default
---

# 三、方法 · 移位集成（DTCWT 平移不变性的廉价替身）

Haar 子带**平移敏感**：1px 平移会把边缘能量在 HL/LH/HH 间乱搬 ⇒ 网络可"放错子带"骗过 L1。
对 $|\mathcal S|$ 个二进位移偏移分别算高频损失再平均，使目标近似平移不变：

$$\mathcal{L}_{\text{shift}}=\frac{1}{|\mathcal{S}|}\sum_{s\in\mathcal{S}}\mathcal{L}_{\mathrm{HF}}\bigl(\mathrm{roll}(\hat y,s),\ \mathrm{roll}(y,s)\bigr),\quad \mathcal{S}=\{(0,0),(1,0),(0,1),(1,1)\}$$

另实现**真·双树复小波 DTCWT**（两棵树 $1$px 偏移 → 复数方向子带取模）；但在强基座上 **Haar + 移位集成** 反而更好。

<v-callout type="success">
实测：移位集成把 **MUSIQ 顶到全项目最高 56.02（非-TTA）/ MANIQA 0.3534** —— 机制层面唯一仍在涨点的项。
</v-callout>

---
layout: default
---

# 三、方法 · Muon：把"正交"从损失推广到优化器

对 $\ge 2D$ 权重维护动量缓冲 $G$，用 **Newton–Schulz 迭代**做近似正交化（对 $G=U\Sigma V^\top$ 逼近 $UV^\top$）：

$$O=\mathrm{NS}(G,5),\qquad \Delta\theta=-\eta\cdot \mathrm{rms}(G)\cdot O$$

- 直觉：正交化使各奇异方向**步长均衡**（谱归一化），小数据 SR 上比 AdamW 更近最优步。
- $1D$ 参数（bias/norm）仍用 AdamW 复合；**两路各自独立 cosine**（`--muon-aux-lr` 解耦）。
- 实测（bs=128 对照）：Muon 在 $1\text{e-}3\sim1\text{e-}2$ **全 lr 段三指标胜 AdamW**，最优 $lr=5e\text{-}3$。

<v-callout type="info">
零预训练 + Muon 已 ≈ 预训练 + AdamW ⇒ **Muon 抵掉一大截预训练收益**。
</v-callout>

---
layout: default
---

# 三、方法 · 预训练与测试时增强

**BSRGAN 预训练**：二阶合成退化在 DIV2K+Flickr2K 预训练 → RealSR 上以 (Muon + 小波 λ8) 微调，四指标同向 +0.09 dB。

**TTA（D4 自集成）**：8 个二面体群变换各前向、反变换回原方向、float 域平均、最后一次性量化：

$$\hat y_{\text{TTA}}=\mathrm{round}\!\Bigl(\tfrac{1}{|G|}\sum_{g\in G} g^{-1}\bigl(f_\theta(g(x))\bigr)\Bigr),\quad |G|=8$$

- 纯推理端免费增益：抬 **SSIM / PSNR / MANIQA**；
- **MUSIQ 常持平**（8 重平均轻微平滑极锐纹理）⇒ 再次说明须**多指标同看**。

---
layout: default
class: text-sm
---

# 四、实验 · 主表①：官方配对协议 A-strict（本项目自测）

<div class="text-xs opacity-70 -mt-1 mb-1">有限量程 Y · 不剃边 · 100 对；**同列最优加粗**、<u>次优下划线</u>。RCAN/SRResNet/RRDB 复现中，完成后并入。</div>

| 方法 | 类别 | 参数 | PSNR-Y↑ | SSIM↑ | MUSIQ↑ | MANIQA↑ |
|---|---|---|---|---|---|---|
| Bicubic | — | — | 31.73 | 0.8876 | 39.40 | 0.2780 |
| SwinIR-light（重训） | Transformer | 0.61M | 32.88 | 0.9026 | 45.43 | 0.3020 |
| SwinIR-largeish（重训） | Transformer | 3.96M | 32.97 | 0.9058 | 46.79 | 0.3050 |
| E11 Mod-SwinIR+Align | Transformer | 4.05M | 33.47 | 0.9143 | 49.15 | 0.3090 |
| EDSR-baseline（复现） | CNN | 1.19M | 33.60 | <u>0.9167</u> | 53.14 | 0.3273 |
| latent rectified-flow（我们） | 生成式 | 32.9M | 28.36 | 0.7900 | 50.12 | 0.2240 |
| s1_b64（AdamW, 零预训练） | U-Net | 18.7M | <u>34.11</u> | <u>0.9246</u> | <u>55.22</u> | <u>0.3420</u> |
| **Ours（完整配方 + TTA）** | U-Net stride-1 | 18.7M | **34.41** | **0.9285** | **55.96** | **0.3534** |

<div class="text-sm mt-2" style="color:#b03024">
相对重训 SwinIR-largeish：<b>+1.44 dB · +0.023 SSIM · +9.2 MUSIQ · +0.048 MANIQA</b>（同协议全指标领先）。
</div>

---
layout: image
image: /figs/fig_results.png
---

---
layout: default
class: text-xs
---

# 四、实验 · 主表②：文献同协议大表 B-128/512 ×4（全方法·全指标）

<div class="text-xs opacity-70 -mt-1">统一 RealSR 裁块协议（更难，输入再退化）；出处 TinySR(2508.17434)。各列 **最优加粗**、<u>次优下划线</u>。</div>

| 方法 | 类别 | PSNR-Y↑ | SSIM↑ | MUSIQ↑ | MANIQA↑ |
|---|---|---|---|---|---|
| BSRGAN | GAN | 26.38 | 0.7651 | <u>63.28</u> | 0.5425 |
| Real-ESRGAN | GAN | 26.65 | 0.7603 | 60.45 | 0.5507 |
| LDL | GAN | 25.28 | 0.7565 | 60.92 | 0.5494 |
| DiffBIR | 扩散 | 25.93 | 0.6525 | 65.66 | <u>0.6296</u> |
| StableSR | 扩散 | 28.04 | 0.7454 | 58.53 | 0.5603 |
| SeeSR | 扩散 | <u>28.14</u> | 0.7712 | 64.74 | 0.6022 |
| ResShift | 扩散 | **28.69** | **0.7874** | 52.40 | 0.4756 |
| SinSR | 一步扩散 | **28.38** | 0.7499 | 55.03 | 0.4904 |
| OSEDiff | 一步扩散 | 27.92 | <u>0.7836</u> | 64.69 | 0.5898 |
| AdcSR | 一步扩散 | 28.10 | 0.7726 | <u>66.26</u> | **0.5927** |
| TSD-SR | 一步扩散 | 27.77 | 0.7559 | **66.62** | 0.5874 |
| TinySR | 一步扩散 | 27.48 | 0.7459 | 65.36 | 0.5804 |

<div class="text-sm mt-1" style="color:#b03024">注：与表①跨协议不可直接横比；生成式以感知换保真，正是下一张图的权衡。</div>

---
layout: default
---

# 四、实验 · 免训练实测的数据集上限（三堵墙）

<div class="text-sm opacity-70 -mt-1 mb-1">工具 <code>diffusion/oracle.py</code>：不训练、不外推，直接在 100 张 GT 上量。</div>

| 误差项 | ×2 | ×3 | ×4 |
|---|---|---|---|
| $E_{\text{align}}$ 亚像素配准 | 0.2px→41.11 · **0.3px→37.69** · 0.5px→33.41 | 同 | 同 |
| $E_{\text{null}}$ 带宽截断 | **40.65** | 34.55 | 31.27 |
| $E_{\text{noise}}$ 传感器（MAD $\hat\sigma\approx0.9/255$） | $\mathrm{PSNR}=10\log_{10}\tfrac{255^2}{\sigma^2}=49.05$ | 49.05 | 49.05 |
| **合成（带宽+噪声）** | **40.1** | 34.4 | 31.2 |
| 合成 + $0.3$px 配准 | **35.7** | ~32 | ~29.5 |

<v-callout type="warning">
×2 上限由**配准**主导 ≈ **34.4–35.7 dB**；我们 **34.41 / 0.9285** 已在带内。修正"38–40dB 噪声底"之说：此数据噪声很小（$\sigma\approx0.9$），真正的墙是配准、其次带宽。
</v-callout>

---
layout: image
image: /figs/fig_ceiling.png
---

---
layout: image-right
image: /figs/fig_tradeoff.png
class: text-sm
---

# 四、实验 · 感知-失真权衡（同协议自证）

同一测试集、同一协议，我们两条臂：

- 判别式回归 **34.41 dB / 0.9285 SSIM**（保真前沿）
- 生成式 latent-flow **28.36 dB / 0.790 SSIM**（以约 **6 dB** 保真换无参考 IQA）

与 Blau & Michaeli（CVPR 2018）完全吻合：

$$\text{追求感知真实}\ \Rightarrow\ \text{必须注入方差}\ \Rightarrow\ \text{MSE 严格上升}$$

→ 真实超分上：**保真被物理封顶，真实感被 390 对数据量封顶**。

---
layout: default
class: text-sm
---

# 四、实验 · 消融（同协议单因子，Δ PSNR-Y）

| 改动 | Δ PSNR-Y | 维度 | 备注 |
|---|---|---|---|
| stride-2 → **stride-1** | **+0.38** | 架构 | 本工作最大单项杠杆 |
| + **小波高频损失 λ8** | +0.11 | 损失 | MUSIQ/MANIQA 同升 |
| + **Muon 5e-3** | +0.15 | 优化器 | 抵一部分预训练 |
| + **移位集成** | +0.02 | 损失 | MUSIQ 最高 56.0 |
| + BSRGAN 预训练 | +0.09 | 数据 | 6× 训练时长 |
| + TTA | +0.10 | 推理 | 8× 推理，SSIM/MANIQA↑ |
| 容量 5M→72.5M | +0.05 | 规模 | 饱和 |

机制项（架构 + 正交频率损失 + 优化器）主导；纯堆参数饱和。

---
layout: image
image: /figs/fig_ablation.png
---

---
layout: default
class: text-sm
---

# 四、实验 · 跨尺度迁移与 scale-adaptive λ

| 尺度 | plain | ours(最优 λ) | 该尺度上限 | 判定 |
|---|---|---|---|---|
| ×2 | 34.11 | **34.41** (λ8+shift+TTA) | ≈35.7 | **已在天花板内** |
| ×3 | 31.04 | **31.13** (λ2) | ≈34.4 | 欠调，仍有空间 |
| ×4 | 29.42 | **29.51** (λ5) | ≈31.2 | 接近天花板 |

最优 λ 随退化强度**反相关**：$\lambda_{\times2}\!\approx\!8,\ \lambda_{\times3}\!\approx\!2,\ \lambda_{\times4}\!\approx\!5$
（×3 处 λ5 过锐致 MUSIQ 反降；×4 λ5 全面有效）。

→ **×3/×4 因本轮欠调仍有余量**，是保真轴上唯一"未做满"的方向。

---
layout: default
---

# 四、实验 · 负结果汇总（28+ 臂，同协议如实报告）

| 路线 | 结论 |
|---|---|
| 扩散 2×2（pixel/latent × flow/reg） | 低于 bicubic / 不超自身零初始化 |
| **Mamba / VSS** 扫描骨干 | 等样本预算 **−2.3 dB**（访存受限、样本效率差）→ 关线 |
| 几何/结构正则：D4 等变、坐标注入、频域路由、门控 FFN、压平金字塔 | 持平或负 |
| 小波架构化：双射小波 U-Net、**双路小波 U-Net**、DTCWT 基 | 均不敌「Haar 高频损失 + 移位集成」 |
| 频域 rectified flow（HF 残差生成） | 修好 bug 仍单调毁图 → 生成不可预测高频=注入伪影 |
| 对抗挖数据多样性（N3，配 Muon） | 中性，未破平台 |

---
layout: default
---

# 五、结论与下一步

- **一套数学上讲得通的配方**：stride-1 + 正交小波高频损失(含移位集成) + Muon + 预训练 + TTA → **34.41 dB / 0.9285 SSIM**，同协议四指标全面领先重训 SwinIR。
- **免训练实测证明**：×2 PSNR 上限 ≈ 34.4–35.7 dB（配准主导），**我们已在天花板内**——不是模型不够，是物理不允许。
- **下一步**：① ×3/×4 用完整配方逼近其上限（唯一明确余量）；② 保真见顶后，**感知轴**（MUSIQ/LPIPS/FID）是主战场。

<div class="text-sm opacity-70 mt-4">复现：README §9 · 测上限：`python -m diffusion.oracle --scale 2` · 本幻灯：`docs/slides/`</div>

---
layout: default
class: text-sm
---

# 参考文献

- Cai et al. **RealSR / LP-KPN**, ICCV 2019. Liang et al. **SwinIR**, ICCV-W 2021. Lim et al. **EDSR**, CVPR-W 2017. Zhang et al. **RCAN**, ECCV 2018.
- Wang et al. ESRGAN ECCV-W 2018 · **BSRGAN** ICCV 2021 · **Real-ESRGAN** ICCV-W 2021.
- Qiao et al. **RealSR-R1**, arXiv:2506.16796 (2025). **TinySR**, arXiv:2508.17434 (2025) — 文献基线表出处。
- **Blau & Michaeli**, The Perception-Distortion Tradeoff, CVPR 2018 (DOI 10.1109/CVPR.2018.00652).
- **Baker & Kanade**, Limits on Super-Resolution..., IEEE TPAMI 2002 (DOI 10.1109/TPAMI.2002.1033210).
- **Robinson & Milanfar**, Fundamental Limits in Image Registration, TIP 2004；Statistical Performance Analysis of SR, TIP 2006.
- **Chatterjee & Milanfar**, Is Denoising Dead?, TIP 2010. Donoho & Johnstone, Wavelet Shrinkage (MAD 噪声估计), Biometrika 1994.
