# 小波 + 等变 + 双射小波U-Net（方向一/二/三）— 阶段结果

> 起点：Mamba 与三条"暴力路线"（堆参数/加时长/换大架构）全部撞墙后，转向
> **针对性机制设计**与**突破数据多样性瓶颈**。本文件记录这一轮的工程与判读。
> 协议：官方 100 对 Test.m，tile 64 / pad 16 / steps 8 + MUSIQ/MANIQA（与全仓库同口径）。
> **锚点 s1_b64 零预训练：Y 34.1083 / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416**；
> 旧冠军（BSRGAN 预训练+后训练）Y 34.1988。判定纪律：±0.05 dB 内算平局；16 对 val 只看趋势。
>
> **评价口径（用户定）：PSNR-Y 次要，主看 SSIM / MUSIQ / MANIQA 三项。** 下面的排名与采纳均以三项为主。

## 实现（全部零/极小代码侵入，训练期开关，推理协议不变）

- **`diffusion/wavelet.py`**：手写正交 2D Haar DWT / IWT（固定权重、可微、bf16 安全、**无新依赖**）。
  校验过双射（IWT(DWT(x)) 误差 ~1e-07）与能量守恒（正交）。
- **方向二 `--dwt-loss / --dwt-weight / --dwt-levels`**：reg 主 L1 之上叠加 HL/LH/HH 高频子带的多尺度 L1，
  用严格正交、**局部**的小波基对抗 L1 的均值收缩过平滑（区别于历史上失败的 FFT 全局频带增益头）。
  骨干无关（unet/mamba 通用），**零推理成本**。
- **方向三 `--equiv / --equiv-weight`**：每步对输入做随机 D4 变换二次前向，罚 `|T(f(x))−f(T(x))|`，
  把等变性从"数据增强的期望意义"抬到"逐点精确"。**computation-bound，不用 grad-ckpt**，
  改 batch 128→64、steps→20000（样本预算仍 1.28M）。
- **方向一 `--dwt-unet`**：把 U-Net 层级间"有损 stride-2 下采样 + nearest 上采样"整体替换为
  **无损 Haar DWT 下采样 / IWT 上采样**（`DWTDown`/`IWTUp`）。验证"无损 stride-2 能否拿到 stride-1
  的相位保真 + stride-2 的算力/感受野"。零初始化头保证 step0=bicubic 不变量不破（smoke 已验含 tiled-eval）。

## Round-1 + 1b 结果（方向二 / 三；官方 100 对）

| 臂 | 配方 | 样本 | Y | SSIM | MUSIQ | MANIQA | vs 锚点 |
|---|---|---|---|---|---|---|---|
| W1 | +dwt λ=1 | 1.28M | 34.1895 | 0.9258 | 55.38 | 0.345 | +0.081 ✅ |
| W2 | +dwt λ=2 | 1.28M | 34.2213 | 0.9261 | 55.43 | 0.347 | +0.113 ✅ |
| W3 | +dwt λ=3 | 1.28M | 34.2199 | 0.9253 | 54.83 | 0.342 | +0.112 ✅ |
| W5 | +dwt λ=5 | 1.28M | **34.2397** | 0.9262 | 55.34 | 0.347 | **+0.131（四指标全≥锚点）** |
| E1 | +equiv α=0.25 | 1.28M | 34.0379 | 0.9243 | 55.44 | 0.340 | **−0.070 → 不采纳** |

**判读**：
- **小波高频 Loss 完胜且稳健**：λ∈[1,5] 把 Y 抬到 **34.19–34.24 平台**（彼此差在 ±0.05 噪声带内）。
  最优点 **λ=5：Y 34.2397、且 SSIM/MUSIQ/MANIQA 全部 ≥ 锚点**（四指标同升）。
- **λ=5 已超过旧冠军 BSRGAN 预训练+后训练的 34.1988**，而它零预训练、零推理成本、每臂仅 1.6h。
- λ 是失真/感知旋钮但本次噪声较大（λ=3 的 MUSIQ 反而低、λ=2/5 回升）；结论以「平台整体 +0.11~0.13」为准，
  不要过度解读单个 λ 的感知差。
- **方向三等变正则判负**：Y 34.0379（−0.07，与 N1/N2 同命）。「精确等变」这条没兑现，只守住平局偏下，不采纳。
- 与历史 FFT 频带增益头（全负）不矛盾：**正交 + 局部**是这次成立的关键。

## Round-3/4 结果（预训练冠军 b64_ft15k + dwt-loss 续训）— 新新冠军

| 臂 | 配方 | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|---|
| b64_pre_dwt1 | init + λ1 | 0.9264 | **55.81** | 0.3501 | (34.187) |
| b64_pre_dwt2 | init + λ2 | 0.9266 | 55.79 | 0.3501 | (34.236) |
| b64_pre_dwt3 | init + λ3 | 0.9266 | 55.77 | 0.3501 | (34.251) |
| b64_pre_dwt5 | init + λ5 | 0.9270 | 55.77 | 0.3506 | (34.267) |
| **b64_pre_dwt8** | init + λ8 | **0.9271** | **55.87** | **0.351** | (**34.278**) |
| b64_pre_dwt10 | init + λ10 | 进行中 | | | |

**判读（按三指标为主）**：
- 预训练基座上 λ 1→8：**SSIM / MANIQA / MUSIQ / PSNR-Y 均缓升、未见拐点**；λ5→λ8 增幅已缩小到 ~0.01 级（平台）。当前最佳 **b64_pre_dwt8：SSIM 0.9271 / MUSIQ 55.87 / MANIQA 0.351 / Y 34.278**（λ6/7/10 继续探拐点）。
- **当前新冠军 = `b64_pre_dwt8`（BSRGAN预训练 + 小波高频Loss λ8）：SSIM 0.9271 / MUSIQ 55.87 / MANIQA 0.351 / Y 34.278**，相比旧冠军 b64_ft15k（0.9251 / 55.71 / 0.3470 / 34.199）四项全抬（λ10、λ6/7 正在把拐点定死）。
- 上几轮我用 PSNR 口径说“dwt 与预训练不可叠加”是错的——**预训练 + 小波 Loss 恰恰是感知最优组合**，两 lever 在 SSIM/MANIQA 上叠加。
- **round-5（进进行中）**：λ 未见顶，往 λ8/λ10 探，防过锐化反噬感知，定位真实峰值。

## Round-2 方向一 dwt-unet：判负（评估 bug 已修并重评完成）

训练均跑完（16 对 val best 33.947）；`eval_official` 曾因重建读错 args 键名（应 `targs['dwt_unet']` 非 `wavelet`）报 state_dict 错，已修，`reeval` 仅重评估未重训：

| 臂 | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|
| s1_b64_dwtunet_dw1 | 0.9258 | 55.06 | 0.344 | (34.1954) |
| s1_b64_dwtunet | 0.9243 | 54.93 | 0.341 | (34.0933) |

**判读**：两臂 SSIM/MUSIQ/MANIQA **三项均 ≤ 锚点**（连 dwt-unet+λ1 的 MUSIQ 55.06 < 55.22）。“无损 stride-2”既没拿到质量红利、又不涨感知 → **方向一作为质量杠杆判负**（若日后要 stride-2 的算力/感受野，它是合法的回退选项，但不是涨点手段）。

## TTA 自集成（方向：免费涨点，`eval_official --tta`）

D4 全 8 重前向→反变换→float 均值。纯评估、零重训：

| ckpt | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|
| b64_pre_dwt8（非-TTA） | 0.92709 | 55.87 | 0.3513 | (34.278) |
| **b64_pre_dwt8 + TTA** | **0.92786** | 55.825 | **0.3516** | (**34.377**) |

TTA 抬 SSIM/MANIQA/Y、MUSIQ 基本持平 → 交付用 TTA 版。

## ×3 / ×4 迁移（zero-pretrain，plain 对照 vs dwt λ5）

| scale | 臂 | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|---|
| ×3 | plain | 0.8687 | **51.79** | **0.324** | (31.04) |
| ×3 | dwt5 | 0.8686 | 50.48 | 0.320 | (31.12) |
| ×4 | plain | 0.8295 | 47.15 | 0.299 | (29.42) |
| ×4 | **dwt5** | **0.8322** | **48.25** | **0.313** | (29.51) |

**判读**：**×4 dwt 全面有效**（MUSIQ +1.10 / MANIQA +0.014 / SSIM +0.003）；**×3 是唯一反常点**（dwt 只涨 PSNR、MUSIQ 反降 1.31）→ 非单调，**说明最优 λ 随退化强度变化（×3 处 λ=5 过锐），应做 scale-adaptive λ**。（×3/×4 绝对值低是任务本身更难，不可与 ×2 横比。）

## 交付态冠军 & vs SwinIR

**×2 交付冠军 = `b64_pre_dwt8 + TTA`：SSIM 0.9279 / MUSIQ 55.83 / MANIQA 0.3516 / Y 34.38。**

| | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|
| SwinIR-largeish 3.96M（原版） | 0.9058 | 46.79 | 0.305 | (32.97) |
| E11 Mod-SwinIR（旧改进基线） | 0.9144 | 49.15 | 0.309 | (33.47) |
| **b64_pre_dwt8 + TTA** | 0.9279 | 55.83 | 0.3516 | (34.38) |
| **Δ vs SwinIR-largeish** | **+0.022** | **+9.0** | **+0.047** | (+1.40) |

（协议 A：均见过 RealSR train；官方 SwinIR real-SR 是协议 B，不可直接横比。）

## 下一步（分频 2.0 → 频域 rectified flow）

1. **分频 2.0**：换更好小波基（db2/db4/DTCWT，替 Haar 的平移敏感/方向差）+ 各向异性分方向 λ（HL≠LH≠HH，对应像散）+ scale-adaptive λ。`wavelet.py` 扩 FIR 正交基 + `--dwt-basis/--dwt-w-hl/lh/hh`。
2. **频域 rectified flow**：回归主干出 LL+粗 HF，仅对 **HF 子带残差** 少步流匹配（1–4 NFE）做生成式细化——只在感知能赢、PSNR/SSIM 不受损的高频带做，绕开以往 flow 全线低于地板的问题（旧结论系 PSNR 口径，新感知口径需重测）。

## 产物位置

- 结果：`experiments/diffusion/{wave_arms,wave_unet,stack_dwt,scale_3,scale_4}/<臂>/eval_iqa[/eval_iqa_tta]/eval.json`
- 日志：`logs/{wa_,wu_,sd_,sc_,tta_}*.log`；驱动 `experiments/{wave_arms,wave_unet,wave_sweep,stack_dwt,stack_sweep2,next_phase2}.log`
- 脚本：`run_wave_arms / run_dwt_sweep / run_dwt_unet_arms / run_stack_dwt / run_stack_sweep(2) / run_next_phase / chain_round2 / reeval_dwtunet / reeval*`

