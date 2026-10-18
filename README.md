# WaveSR：真实相机双焦段超分辨率

单反实拍的双焦段（wide / tele）配对数据，本质是一个**有物理上限的逆问题**：
两个焦段之间存在 0.3–0.5 像素的非刚性局部视差、镜头点扩散、以及测光微漂移。
本仓库围绕这一个问题，记录了 4 轮完整的「立项 → 证伪 → 换路线 → 收敛」。

最终交付：

| | |
|---|---|
| **模型** | Stride-1 残差 U-Net（b64，18.67M）+ 小波高频损失 + Muon + 移位集成 |
| **后处理** | WD-DCP：免训练、只修低频子带的数据一致性投影 |
| **读数** | 官方 100 对 **Y 34.4632 / SSIM 0.9284**（含 8 折 TTA），单次前向 34.3800 |
| **对照** | SwinIR-L 官方重训 33.9707；bicubic 地板 31.7335 |
| **物理上限** | 配准墙给出 34.4–35.7 dB（`diffusion/oracle.py` 实测） |

全部数字的唯一底账：[`docs/results/RESULT_LEDGER.md`](docs/results/RESULT_LEDGER.md)。

---

## 目录

- [一、历史脉络（按时间读）](#一历史脉络按时间读)
- [二、方法的四块积木](#二方法的四块积木)
- [三、与官方基线的对比](#三与官方基线的对比)
- [四、物理上限：为什么 34.4 就到底了](#四物理上限为什么-344-就到底了)
- [五、被关掉的线路](#五被关掉的线路)
- [六、评测协议与指标体系](#六评测协议与指标体系)
- [七、复现](#七复现)
- [八、仓库结构](#八仓库结构)
- [九、已知的坑](#九已知的坑)

---

## 一、历史脉络（按时间读）

### 第 0 阶段：从上游工具链开始（2021 → 2026-09）

仓库最初是 RealSR / KAIR 工具链的 fork（上游 README 存于
[`exp/archive/README.upstream.md`](exp/archive/README.upstream.md)）。
2026-09-13 起，第一次在自己的数据上跑基线。

### 第 1 阶段：SwinIR（09-13 ~ 09-14）—— 先建立「尺子」

**问题**：真实配对超分到底能到多少？判别式架构里谁最好？

**做法**：在 Mod-SwinIR 上加训练期对齐损失（Align-L1），并试容量对齐与轻量化变体。

| 实验 | PSNR-Y | 判定 |
|---|---:|---|
| **E11：Mod-SwinIR-base + Align-L1（4.05M）** | **33.4655** | **阶段锚点** |
| E2：容量对齐版 | 32.9735 | 负 |
| E1：轻量版 | 32.8827 | 负 |

**收获**：两个东西留了下来 ——
① **评测协议**（BT.601 limited-range Y、uint8、modcrop4、不裁剪、100 对）；
② **锚点 E11 = 33.47 dB**。此后每个新骨干的第一句话都是「比 E11 高多少」。

### 第 2 阶段：扩散（09-16 ~ 09-17）—— 一整条线被证伪

**动机**：判别式回归在视差处会把多个可能的边缘位置求平均 →
条件均值模糊。生成式（flow matching）理论上能采到其中一个模态而不糊。

**做法**：建了 latent / pixel × flow / reg 的 2×2 矩阵，再补 latent-XS 与 res-VAE 两臂。

| 臂 | PSNR-Y | SSIM-Y | 步数 | 判定 |
|---|---:|---:|---:|---|
| **pixel reg（DiT-S 34M）** | **33.7627** | 0.9191 | 1 | 超过 E11 |
| latent reg（Flux） | 31.1234 | 0.8836 | 1 | 低于 bicubic 31.73 |
| latent flow（res VAE） | 29.4617 | 0.8293 | 40 | 崩 |
| latent flow（Flux） | 28.3614 | 0.7901 | 40 | 崩 |
| latent flow（Flux-XS） | 27.0532 | 0.7359 | 40 | 崩 |
| **pixel flow** | **26.7913** | 0.5504 | 16 | **完全失败** |

**判词**：生成式在这里不是「差一点」，是差 6–7 dB。根因两条：
① 扩散采样的随机漂移破坏像素级对齐（Blau–Michaeli 下界在起作用的直接体现）；
② 训练集只有 390–406 对，生成式先验根本训不出来。

**这条线的价值是反面的**：它用一整周的算力证明了「在这个数据量上别碰 flow」。
第 4 阶段的 HF-flow 复核又独立确认了一次。

### 第 3 阶段：换骨干（09-17 ~ 09-19）—— 找到真正的杠杆

**问题**：pixel reg 已经 33.76，但像素级 DiT 每步 0.95 s、34M 参数。能不能又快又好？

**做法**：写一个接口兼容的 U-Net（`--backbone unet`），然后做两组对照：

| 对照 | 臂 | 参数量 | PSNR-Y |
|---|---|---:|---:|
| stride-2（在 LR 尺度算） | U-Net base64 | 18.68M | 33.56 |
| **stride-1（在 HR 尺度算）** | U-Net base32 | **4.99M** | **33.9445** |
| stride-1 | base40 | 7.58M | 34.0070 |
| stride-1 | base48 | 10.73M | 33.9946 |
| stride-1 | base64 | 18.67M | **34.1083** |

**两条结论，一直用到最后：**

1. **`--native-lr 0` 是全项目最大的单项杠杆（+0.38 dB）。**
   4.99M 的 stride-1 压过 18.68M 的 stride-2，还压过 34M 的 DiT。
   机理：bicubic 放大后 HR 网格上仍有真实高频需要重建，
   先下采样到 LR 再 pixel-shuffle 抬回来，等于把要重建的相位信息丢了。
2. **容量在 5M 之后完全饱和。** 4.99 / 7.58 / 10.73M 三点落在 ±0.03 dB 内。

**附带**：预训练（BSRGAN 退化）只值 +0.09 dB（34.1083 → 34.1988）。

### 第 4 阶段：小波损失与优化器（09-22 ~ 09-25）—— 在正确的骨架上加机制

骨架定了，剩下的是监督信号。**空间 L1 对高频是「均摊」的**：
它不区分「结构位置错了」和「纹理细节不够」，而后者才是超分真正要补的。

**① 小波高频损失**：把 Haar 分解的 HL/LH/HH 三个子带单独做 L1，LL 完全剔除。

| λ_DWT | PSNR-Y | SSIM-Y | MUSIQ |
|---:|---:|---:|---:|
| 1 | 34.1895 | 0.9258 | 55.38 |
| 5 | 34.2397 | 0.9262 | 55.34 |
| **8** | **34.2781** | 0.9271 | **55.87** |
| 10 | 34.2876 | 0.9270 | 55.80 |

**② 移位集成**：Haar 没有平移不变性 → 损失里对 ±1 像素的 4 个二进位移求平均。
**+0.03 dB / +0.15 MUSIQ，零推理成本**，是全项目性价比最高的一行代码。

**③ Muon 优化器**：用 Newton-Schulz 正交化替代 AdamW 的标量更新。

| 优化器 | lr | PSNR-Y | MUSIQ |
|---|---:|---:|---:|
| AdamW | 1e-4 | 34.2417 | 55.46 |
| Muon | 1e-3 | 34.3006 | 55.98 |
| **Muon** | **2e-3 ~ 5e-3** | **34.39** | **55.8–56.0** |

**所有学习率点的 Muon 都赢 AdamW**，最佳区 2e-3 ~ 5e-3。
（注意这组 A/B 没开移位集成，所以绝对水位低于最终 champion，看的是**优化器之差**。）

**④ 顺路证伪的一批**：db2 / db4 / DTCWT 基底（Haar 已饱和）、
双分支小波 U-Net（−0.4 dB）、CoordConv 与频域硬路由（< 0.03 dB）。

### 第 5 阶段：官方基线阶梯（09-26）—— 把「赢多少」变成可核对的数字

**问题**：自己训练的基线不算数。必须用**官方仓库原码**、**同配方**、**同协议**重训。

做法：`third_party/` 原码 + `diffusion/baselines_official.py`，
8+ 臂统一 AdamW 1e-4 + L1 + EMA、样本上限 1.28M、100 对 / tile 64 / pad 16。

| 臂 | 参数量 | PSNR-Y | +TTA |
|---|---:|---:|---:|
| Bicubic | — | 31.7335 | — |
| EDSR-baseline | 1.37M | 33.4946 | 33.5300 |
| HAT-L | 40.70M | 33.5457 | 33.5931 |
| VDSR | 0.59M | 33.6205 | 33.6780 |
| RCAN | 15.44M | 33.7818 | 33.8488 |
| SRFormer | 10.40M | 33.8724 | 33.9989 |
| RDN | 22.12M | 33.9061 | 33.9810 |
| SwinIR-L | 11.75M | 33.9707 | 34.0241 |
| **`unet_ctrl`（我们的骨干 + 基线配方）** | 18.67M | **34.0339** | 34.1664 |
| **Ours（全配方）** | 18.67M | 34.3096 | **34.4063** |

**两条能写进论文的读数：**

1. **剥掉所有外挂，骨干仍然第一。**
   `unet_ctrl` 只有基线配方（无预训练 / 无小波 / 无 Muon / 无移位），
   34.0339 超过所有判别式基线的最好成绩（SwinIR-L 33.9707）。
   → **冠军优势来自架构，不是来自损失工程或优化器。**
2. **最大的模型输了。** HAT-L 40.7M 只有 33.5457，比 11.75M 的 SwinIR-L 低 0.43 dB，
   甚至不如 0.59M 的 VDSR。这是第 3 阶段「容量饱和」结论在别人的架构上的复现。
   诚实说明：各臂共享的是样本**上限** 1.28M，不是同等**消耗**
   （HAT-L 只跑完 0.20M、SwinIR 1.06M，其余满额，共享早停规则）。

### 第 6 阶段：后处理与扩容（09-26 ~ 10-15）—— 收尾

**① WD-DCP（免训练数据一致性投影）**：网络输出后做一次
「模糊 → 下采样 → 与 LR 比对 → 上采样回」的修正，但**只用修正量的 LL 低频子带**，
HL/LH/HH 用 α=0 完全冻结。这样相机视差、测光漂移被修掉，
网络生成的高频纹理一个像素都不动（数学证明见 [§二.4](#4-wd-dcp免训练diffusioneval_officialpy---dcp-1)）。

| 方案（p_shift4） | PSNR-Y | SSIM-Y | MUSIQ |
|---|---:|---:|---:|
| 原始单次前向 | 34.3096 | 0.9277 | **56.02** |
| + WD-DCP（η=0.12） | 34.3800 | 0.9277 | 55.51 |
| + TTA | 34.4063 | **0.9285** | 55.96 |
| **+ WD-DCP + TTA** | **34.4696** | 0.9281 | 55.05 |

步长 η 的保真峰值在 0.15（34.3916），η=0.12 是八维指标最均衡的点。
**保真与自然感在这里是对立的**：每 0.07 dB 的 PSNR，要付 ~0.5–0.9 的 MUSIQ。

**② 跨架构验证**：把 WD-DCP 直接插在官方 SwinIR-L 上（4 对子集），
PSNR 33.3378 → 33.3517，同时 LPIPS / DISTS / NIQE 三个感知指标同步变好。
免训练、零微调、在完全不同的架构上也是正的。

**③ Scale-96（41.19M）**：把 base 提到 96，batch 拉到 240 吃满 46.7 GB 显存。

| 配置 | PSNR-Y | MUSIQ | MANIQA |
|---|---:|---:|---:|
| Scale-96 原始 | 34.3739 | **56.14** | **0.3545** |
| Scale-96 + WD-DCP | 34.4473 | 55.61 | 0.3483 |
| **Scale-96 + WD-DCP + TTA** | **34.5419** | 55.52 | 0.3482 |

**参数 2.2 倍、算力 2.2 倍、batch 1.9 倍，只换 +0.07 dB。**
这是第三次（也是最后一次）确认容量饱和 —— 这条路线已经不是靠堆模型能推的了。

---

## 二、方法的四块积木

### 1. Stride-1 残差 U-Net（`diffusion/unet.py`）

```
LR ──bicubic×2──> 2H×2W ──stem(3×3,s=1)──> [ResBlock×2 @base]
                                  │
                                  ├─stride2─> [Res×2 @2b]
                                  │                 │
                                  │                 ├─s2─> [Res×2+Attn @4b]
                                  │                 │            │
                                  │                 │            └─s2─> [Res×2+Attn @4b]  ← bottleneck
                                  └────── skip ─────┼─────────────────────┘
                                                    └ 逐级 concat + 上采样
                                                              ↓
                          零初始化 3×3 输出头 ──> Δy ──> ŷ = bicubic(LR) + Δy
```

- **输入侧不下采样**：stem 直接在 2H×2W 上算（这是 +0.38 dB 的来源）。
- **注意力只放在阶段 2、3**：全分辨率上 token 数 4096，
  n² attention 会占掉 64% 的算力却伤局部细节。
- **零初始化**：每个 ResBlock 的第二个卷积、注意力输出投影、网络输出头全部零初始化。
  step 0 时误差恒等映射，输出严格等于 bicubic（31.7335）。
- **可选 native-LR 模式**（`--native-lr 1`）：stem 用 stride-2 降到 LR 尺度跑，
  最后 pixel-shuffle 抬回。省 29% FLOPs、**显存降 4.5 倍**，但代价是 −0.38 dB。
  **只对回归有效**：flow 的 ODE 状态在 HR 带白噪声，下采样会混叠掉速度场要重建的信息。

### 2. 小波高频损失（`diffusion/wavelet.py`）

DWT(x) = [x_LL, x_HL, x_LH, x_HH]

L_DWT = (1/J) Σ_j (1/3) Σ_{s∈{HL,LH,HH}} ‖DWT_j(ŷ)_s − DWT_j(y)_s‖₁

总目标：L = ‖ŷ − y‖₁ + λ · L_shift_DWT，λ = 8（x2 尺度）。

三个设计点，每个都被消融验证过：

| 设计 | 不这么做会怎样 |
|---|---|
| 剔除 LL 子带 | 与空域 L1 重复加权，低频被惩罚两次，测光漂移变成梯度 |
| 只做高频 | 直接把 λ 加到全频段，等于放大 L1，没有解耦 |
| 二进位移平均 | Haar 无平移不变性 → 同一纹理随位置得到不同梯度，λ=8 时会过冲 |

### 3. Muon 优化器（`diffusion/train_pixel.py --optimizer muon`）

X₀ = G/‖G‖_F，X_{k+1} = ½ X_k (3I − X_kᵀ X_k)

5 步 Newton-Schulz 迭代得到梯度矩阵的最接近正交极分解。
效果是各层更新步长的奇异值分布变均匀，不再被少数主导方向吃掉。
**在 10k 步同预算下稳定 +0.15 dB。**

### 4. WD-DCP（免训练，`diffusion/eval_official.py --dcp 1`）

1. Δ_k = η · Up( Down(x_k ⊛ G_σ) − y_LR )
2. DWT(Δ_k) = [Δ_LL, Δ_HL, Δ_LH, Δ_HH]
3. Δ̃_k = IDWT([Δ_LL, α·Δ_HL, α·Δ_LH, α·Δ_HH])，取 **α = 0**
4. x_{k+1} = clip(x_k − Δ̃_k, 0, 1)

α=0 时，因为 DWT 是正交变换，DWT(Δ̃_k) 的 HF 分量恒为 0，
所以 x_{k+1} 与 x_k 的 HL/LH/HH 子带**逐元素恒等**（在未触发截断的线性区间内）。
**网络生成的高频一个像素都没动，只修低频的视差与测光漂移。**
单元测试：`scripts/test_wd_dcp.py`，实测高频差值 0.000000。

---

## 三、与官方基线的对比

完整八维指标表见 [`docs/results/RESULT_LEDGER.md`](docs/results/RESULT_LEDGER.md)。
这里只给 PSNR-Y 与关键感知指标（官方 100 对，A-strict 协议）。

### 3.1 单次前向

| 方法 | 参数 | PSNR-Y | SSIM-Y | MUSIQ | MANIQA |
|---|---:|---:|---:|---:|---:|
| Bicubic | — | 31.7335 | 0.8876 | 39.40 | 0.2782 |
| EDSR-baseline | 1.37M | 33.4946 | 0.9149 | 51.49 | 0.3208 |
| HAT-L | 40.70M | 33.5457 | 0.9165 | 51.94 | 0.3206 |
| VDSR | 0.59M | 33.6205 | 0.9167 | 52.90 | 0.3175 |
| RCAN | 15.44M | 33.7818 | 0.9199 | 54.02 | 0.3321 |
| SRFormer | 10.40M | 33.8724 | 0.9237 | 54.68 | 0.3391 |
| RDN | 22.12M | 33.9061 | 0.9218 | 54.82 | 0.3395 |
| SwinIR-L | 11.75M | 33.9707 | 0.9224 | 54.58 | 0.3378 |
| `unet_ctrl`（骨干 + 基线配方） | 18.67M | **34.0339** | **0.9240** | 54.96 | 0.3399 |
| **Ours（全配方）** | 18.67M | **34.3096** | **0.9277** | **56.02** | **0.3534** |
| **Ours + WD-DCP** | 18.67M | **34.3800** | 0.9277 | 55.51 | 0.3472 |
| **Ours + WD-DCP + TTA** | 18.67M | 34.4632 | **0.9284** | 55.44 | 0.3470 |
| **Scale-96 + WD-DCP + TTA** | 41.19M | **34.5419** | **0.9294** | 55.52 | 0.3482 |

> MUSIQ / MANIQA 取同一 checkpoint 的 `eval_iqa` 读数；
> LPIPS / DISTS 只对部分臂算过（见底账 CSV 的空值），不要跨行比较。

### 3.2 相对增量（都相对同一 checkpoint）

| 操作 | ΔPSNR | 代价 |
|---|---:|---|
| bicubic → `unet_ctrl`（换架构） | **+0.30** | 一次重训 |
| `unet_ctrl` → Ours（预训练 + 小波 + Muon + 移位） | **+0.28** | 4 项机制叠加 |
| + WD-DCP | +0.07 | 免训练，推理 ~2× |
| + 8 折 TTA | +0.06 | 推理 8× |
| base 64 → 96 | +0.07 | 参数 2.2×，显存 1.5× |

---

## 四、物理上限：为什么 34.4 就到底了

`diffusion/oracle.py` 在 100 对真实数据上做三项免训练的误差分解
（复现：`python -m diffusion.oracle --scale 2`）：

MSE_total = E_registration + E_bandlimit + E_sensor_noise

| 项 | 测量方式 | 上限 |
|---|---|---:|
| **配准墙** | 在真实 HR 上施加刚性亚像素位移 | 0.3 px → 37.7 dB；**0.5 px → 33.4 dB** |
| 频带截止墙 | 理想矩形低通截断到 LR 奈奎斯特 | 40.65 dB |
| 传感器噪声 | Donoho MAD 小波估计 | 49.05 dB |
| **综合** | — | **34.4 – 35.7 dB** |

**结论**：常被当成 40 dB 拦路的「传感器噪声」其实有 49 dB，根本不是问题。
真正卡死的是**双焦段变焦的局部视差**：0.35 px 量级的错位直接把上限压到 34.4–35.7。
我们交的 34.46 已经贴在这个墙上，
再加参数量只能在这 0.1 dB 里打转 —— 这也解释了第 6 阶段 Scale-96 为什么只值 +0.07。

**这同时解释了所有「感知换保真」的现象**：视差让真值边缘呈多模态分布，
任何把分布往一个模态拉的机制（DCP、GAN、flow）都在做同一件事 ——
保真度上去、自然感下来，或者反过来。

---

## 五、被关掉的线路

完整清单见 [`docs/results/RESULT_LEDGER.md`](docs/results/RESULT_LEDGER.md) §8。挑三条最值得记的：

### Mamba / 视觉状态空间

8 臂矩阵、同预算对照，最好读数 **33.4182**，比同预算 U-Net 低 2.28 dB。

| 事实 | 数字 |
|---|---|
| 官核（`mamba_ssm`）MFU | **≥3.6%**（U-Net b64 是 20.0%） |
| 纯 torch 回退 | ~0.1%，且 batch>1 即 OOM |
| 不降采样的步时 | 30 秒/步 → 2 万步要 7 天 |
| 装 nvcc 12.8 + 源码编译 + 写精确回退 + 8 臂矩阵 | **10.3 小时** |
| 同期「加一个频域损失开关」单臂 | **1.6 小时，超过 Mamba 全部臂** |

**质量与工程两条战线全输。**
顺带一个反直觉细节：stride-1 在 Mamba 内部也有效（+0.89 dB，复现了第 3 阶段的杠杆），
但绝对水位还是低 U-Net 约 1 dB —— 说明瓶颈在状态空间范式本身，不在配置。

### 频域残差 flow（HF-flow）

第 2 阶段否定 flow 时用的是整图生成。为了排除「只是整图 flow 不行」，
又在 wavelet 高频子带上做了残差 flow（`hf_flow`）：回归主干 + 高频小残差流。

| refine scale | PSNR-Y | SSIM-Y |
|---|---:|---:|
| 0.10 | 31.0746 | 0.7622 |
| 0.20 | 27.1776 | 0.5414 |
| 0.35 | 23.1929 | 0.3482 |

**单调崩坏，且是 bug-free 的崩坏。** 这一步把「flow 在这数据上不行」
从「猜测」变成了「结论」。

### 最佳感知分在哪

值得注意：**MUSIQ / MANIQA 的最高分是「什么都不改」的原始输出**
（56.02 / 0.3534），不是任何加料版本。
**所有后处理和所有损失工程都在用感知分换保真分。**
如果目标是画质观感，正确做法是关掉 WD-DCP 与 TTA。

---

## 六、评测协议与指标体系

### 6.1 协议（A-strict，全表统一）

| 项 | 规定 |
|---|---|
| 色彩空间 | ITU-R BT.601 **limited-range Y**（16–235） |
| 量化 | 推理后 `clip(round(y×255), 16, 235)` 的 uint8 域 |
| 裁剪 | `modcrop4` 保证整除，**shave = 0**（不裁边缘） |
| 测试集 | 官方全量 100 对（Canon 001–050 + Nikon 001–050） |
| 分块 | tile 64 / pad 16，浮点融合加权累加 |

**两条地板线**：bicubic **31.7335**；bicubic 过一次 Flux VAE 往返 **31.5971**
（= latent 臂零初始化时的输出）。任何 latent/flow 臂低于 31.60，
意思就是「连什么都不做都不如」。

### 6.2 八维指标

| 指标 | 全参考 | 方向 | 用途 |
|---|:---:|:---:|---|
| PSNR-Y | ✓ | ↑ | 保真主指标 |
| SSIM-Y | ✓ | ↑ | 结构 |
| PSNR-RGB | ✓ | ↑ | 色度保真 |
| LPIPS（AlexNet） | ✓ | ↓ | 深层感知 |
| DISTS | ✓ | ↓ | 纹理 |
| NIQE | ✗ | ↓ | 自然度 |
| MUSIQ | ✗ | ↑ | 多尺度画质 |
| MANIQA | ✗ | ↑ | 人类感知 |

**判断规则**：±0.05 dB、感知指标小幅波动 = 平手，不作结论。

### 6.3 口径纪律

- **参数量**必须 `sum(p.numel() for p in net.parameters())`。
  数 `state_dict` 会把 `relative_position_index` 这类 int64 buffer 算进去，
  SwinIR 会被数成 14.41M（真值 11.75M）、SRFormer 16.37M（真值 10.40M）。
- **MFU** 一律「前向 + 反向」口径。forward-only 会系统性低估约 3 倍。
  权威表在 [`docs/INFRA.md`](docs/INFRA.md)。
- **协议标签不得混用**：A-strict（我们的配对协议）/ B-128-512 / C-REdeg / D-blind。
  文献数字只能带标签引用，不能进主表。

---

## 七、复现

### 7.1 环境

```bash
conda create -n wavesr python=3.11 -y && conda activate wavesr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install pyiqa timm opencv-python pywavelets einops scipy
```
（国内机器：`export HF_ENDPOINT=https://hf-mirror.com`，huggingface.co 不通。）

### 7.2 训练交付模型

```bash
python -m diffusion.train_pixel \
  --data-root "data/RealSR/V3" --out experiments/diffusion/wavesr_base64 \
  --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --attn-levels 2,3 \
  --native-lr 0 --objective reg --residual 1 --scale 2 --lr-patch 64 --batch 128 --amp \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 \
  --ema 0.999 --steps 10000 --save-every 2000
```

Scale-96（显存极限版）：`--base 96 --batch 240 --grad-ckpt`，余同。

### 7.3 评估

```bash
# 原始单次前向
python -m diffusion.eval_official --ckpt <ckpt> --mode pixel --objective reg \
  --tile 64 --pad 16 --tile-batch 128 --iqa --out experiments/eval_raw

# + WD-DCP + TTA（交付读数 34.4632）
python -m diffusion.eval_official --ckpt <ckpt> --mode pixel --objective reg \
  --tile 64 --pad 16 --tile-batch 128 \
  --dcp 1 --dcp-eta 0.12 --dcp-sigma 0.6 --dcp-decay 0.0 --tta --iqa --out experiments/eval_deliver
```

### 7.4 上限测定

```bash
python -m diffusion.oracle --scale 2 --data-root "data/RealSR/V3"
```

### 7.5 取回远程结果

```bash
ssh ds@<host> "cd ~/realsr && find experiments -name eval.json | tar -czf - -T -" \
  | tar -xzf - -C .tmp_remote
python docs/results/collect_evals.py     # → docs/results/all_evals.csv
```

---

## 八、仓库结构

```
RealSR/
├── README.md                      # 本文档（唯一入口）
├── diffusion/                     # 核心库
│   ├── unet.py                    # Stride-1 残差 U-Net（含 native-LR、光学条件嵌入）
│   ├── dit.py                     # DiT-S / XS（对照骨干）
│   ├── train_pixel.py             # 训练引擎（Muon / 小波 / 预训练 / 编译）
│   ├── eval_official.py           # 评测引擎（八维指标 / WD-DCP / TTA / 分块）
│   ├── wavelet.py                 # Haar 正逆变换
│   ├── baselines_official.py      # 官方原码加载器（EDSR/RCAN/VDSR/RDN/SwinIR/HAT/SRFormer）
│   ├── baselines_sr.py            # 早期自研经典基线（不作官方引用）
│   ├── oracle.py                  # 三墙误差分解
│   ├── flow.py / train_latent.py  # 扩散线（已关闭，留档）
│   ├── mamba_sr.py                # Mamba 线（已关闭，留档）
│   └── hf_flow.py                 # 频域残差 flow（已关闭，留档）
├── docs/
│   ├── README.md                  # 文档地图
│   ├── results/
│   │   ├── RESULT_LEDGER.md       # 结果总账（唯一底账）
│   │   ├── all_evals.csv          # 156 份 eval.json 的机器可读版
│   │   ├── collect_evals.py       # 重新生成 CSV
│   │   └── logs/                  # 远程日志与每臂 args.json 归档
│   ├── INFRA.md                   # 算力/MFU/显存/排程（单一权威）
│   ├── baseline/                  # 基线目录与综述
│   ├── slides_new/                # 13 页汇报（slidep DSL + PPTX/PDF）
│   └── archive_slidev/            # 旧版 Slidev 幻灯片（已被 slides_new 取代）
├── exp/
│   ├── FINAL_REPORT.md            # 逐次实验编年史
│   ├── METRIC_SUITE_AND_DCP.md    # 评测体系与 WD-DCP 理论
│   ├── DCP_RESULTS.md             # DCP 扫描专报
│   ├── RELATED_WORK_AND_UPPER_BOUND.md
│   └── archive/                   # 27 份被取代的阶段文档（带索引）
├── scripts/
│   ├── test_wd_dcp.py             # WD-DCP 高频零损伤单测
│   ├── test_physics_loss.py       # LL-Swap 与对齐损失单测
│   ├── test_optical_unet.py       # 光学条件嵌入恒等初始化单测
│   └── server/                    # 训练/评测流水线（tmux + 幂等 eval.json）
├── model/                         # 早期 Mod-SwinIR 包（第 1 阶段）
└── data/                          # 数据集（不进仓库）
```

### 数据

- **Train**：406 对（Canon 5D3 + Nikon D810 各半），切片 64×64 LR / 128×128 HR。
- **Test**：100 对（50 + 50），全量评测。
- **解码缓存**：`data/decoded/`，4562 张 / 31.2 GB 预解码 uint8 blob，
  `precache_hr.py` 21 秒建好，`DecodedStore` 按路径 memmap 零拷贝索引。
  建之前 RealSR 装载 10.95 ms/item；建之后瓶颈转移到退化本身（BSRGAN 153 ms/item）。

---

## 九、已知的坑

这些都是真踩过的，写下来省时间。

**架构改动必须同步评估端。** Scale-96 那轮评估有 2/4 段直接崩在
`Unexpected key(s): opt_embed.mlp.*` —— 训练开了 `--optical-cond`，
评估端没构造这个子模块。日志里保留着原始 traceback（`docs/results/logs/scale96.log`）。

**`model.align` 必须用于分块对齐。** `align = in_stride × 2^(levels-1) × out_scale`（=32）。
图片尺寸不是 align 整数倍时要先 pad 再裁回，
RealSR 的 1000 % 32 = 8，不处理会在 skip concat 处崩。

**`tmux has-session -t NAME` 是前缀匹配。** 脚本里写 `while tmux has-session -t baserep`，
而自己叫 `baserep2` → 匹配到自己、永远等自己、死锁。精确匹配要写 `-t '=baserep'`。

**不要 `pkill -f "<脚本名>"`。** ssh 远端那条 `bash -c` 的命令行里就含这个串，
会把自己一起杀掉。用 `[c]hain_afte` 这类自排除模式。

**`eval_official.py` 经 `| tee` 时 stdout 块缓冲**，日志几十分钟不刷新，
别用 `tail` 判进度，看进程存活 + `nvidia-smi`。

**Windows → Linux 的 `.sh` 必须是 LF。** Python `write_text` 会写出 CRLF，推上去 `syntax error`。

**评估成本要提前算。** 像素 flow 的官方评估 9–10 s/NFE/对（NFE16 跑 100 对约 4 h）；
latent flow 只要 ~7 min；回归臂 ~12 min。排程前先按这个量估。

---

## 引用与致谢

数据集与评测协议来自 RealSR（Cai et al., CVPR 2019 NTIRE Workshop）；
判别式基线来自 KAIR / SwinIR（Liang et al., ICCVW 2021）/ HAT / RCAN / SRFormer 官方仓库；
Muon 优化器来自 Keller Jordan 的 Muon。
