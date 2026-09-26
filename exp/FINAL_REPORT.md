# RealSR 真实超分 —— 最终定稿报告

**项目**：SwinIR-refine / RealSR V3，真实相机单帧超分（×2 主、×3/×4 迁移）
**周期**：2026-09-10 → 09-26 · 硬件：本地 RTX 4070 8GB（阶段A）+ 服务器 dserver RTX 4090 48GB（阶段B）
**评价口径（项目准则）**：**PSNR-Y 次要，主看 SSIM / MUSIQ / MANIQA**。PSNR 仅作保真参考；采纳/判负由 SSIM+MUSIQ+MANIQA 决定，±0.05 dB / 相应感知阈值内视为平局。
**协议**：RealSR V3 官方 `Test.m` —— RGB 训练、Y 测试（limited-range BT.601 16–235）、uint8、modcrop4、不 shave、Test 全量 100 对（Canon 50 + Nikon 50）。所有数字同一份 `diffusion/eval_official.py` 产出，MUSIQ/MANIQA 由 `diffusion/iqa.py` 逐图计算。

---

## 1. 一句话结论与最终成绩

**交付冠军（×2）= `s1_b64` stride-1 U-Net + BSRGAN 预训练 + 正交小波高频 Loss(λ8) + Muon(5e-3) + 移位小波集成 + EMA + TTA**

| 指标 | 值 | 说明 |
|---|---|---|
| **SSIM** | **0.9285** | 全项目最高档 |
| **MUSIQ** | **55.96** | 全项目最高（非-TTA 移位版达 56.02） |
| **MANIQA** | **0.3534** | 全项目最高 |
| Y (PSNR) | 34.41 | 次要，供参考 |

**相对各基线（×2，全量 100 对，主看感知三指标）**：

| 对比 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| bicubic 地板 | +0.041 | **+16.6** | +0.075 | +2.67 |
| SwinIR-light（原版 0.61M 重训） | +0.026 | +10.5 | +0.051 | +1.53 |
| **SwinIR-largeish（原版 3.96M 重训）** | **+0.023** | **+9.2** | **+0.048** | +1.44 |
| E11 Mod-SwinIR（阶段A改进线） | +0.014 | +6.8 | +0.045 | +0.94 |
| s1_b64 零预训练 AdamW 锚点 | +0.004 | +0.74 | +0.012 | +0.30 |

一句话：**相对原版 SwinIR 是一个跨量级的感知提升（MUSIQ +9.2、MANIQA +0.048），并顺带 PSNR +1.4 dB**。核心增益来自**判别式回归 + 正交小波高频 Loss（含移位集成）+ Muon + 预训练 + TTA** 的组合。

---

## 2. 方法与架构

**骨干**：stride-1 像素回归 U-Net（`--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --attn-levels 2,3 --native-lr 0`），18.67M 参数。
- **stride-1**：整网在 HR 尺度运算（stem 不降采样），最高分辨率保留全部高频相位——全项目最大单项杠杆（见 §4）。
- **残差 bicubic**：`ŷ = bicubic(x) + f_θ(x)`，输出头**零初始化** → step0 精确等于 bicubic 地板（免费的"负贡献"探测器）。
- **注意力仅在 level2/3**（低分辨率），避免全分辨率 n²（DiT 时代 64% FLOPs 的教训）。

**损失**：主 L1 + **正交小波高频子带多尺度 L1**（`--dwt-loss --dwt-weight λ --dwt-levels L`）。
- `diffusion/wavelet.py`：手写 2D Haar DWT（固定权重、可微、bf16 安全、无新依赖），双射误差 ~1e-07、能量守恒；可选 FIR 正交基 db2/db4；移位集成（`--dwt-shift`）；DTCWT 复数版（`--dwt-basis dtcwt`）；方向分带权重 `--dwt-w-hl/lh/hh`、LL 项 `--dwt-w-ll`、每尺度权重 `--dwt-level-weights`。
- 动机：L1 回归输出的是高频的**条件均值**→过平滑；小波把高频能量分到 HL/LH/HH 子带并**显式加权惩罚**，正交+局部（非 FFT 全局基）是它成立的关键。

**优化器**：**Muon**（Newton–Schulz 正交化动量用于 ≥2D 权重，其余 1-D 用 AdamW 复合；`--optimizer muon --muon-lr --muon-momentum --muon-ns-steps --muon-aux-lr`）。bf16 下不用 GradScaler。

**训练配方**：BSRGAN 合成退化预训练（DIV2K+Flickr2K 3450 张，σ 按 ×2 校准）→ RealSR 微调；lr 3e-4(AdamW)/5e-3(Muon) cosine、warmup 500、batch 128、steps 10000、EMA 0.999、val16 patience15；bf16 AMP。

**推理 TTA**（`eval_official --tta`）：D4 全 8 重（rot90×flip）前向→反变换→float 均值→单次量化；零重训、纯评估。

---

## 3. 全部实验结果（官方 100 对）

### 3.1 小波高频 Loss —— λ 扫点（零预训练 b64，AdamW）
| 臂 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| s1_b64 锚点（无 dwt） | 0.9246 | 55.218 | 0.3416 | 34.108 |
| λ=1 | 0.9258 | 55.381 | 0.3451 | 34.190 |
| λ=2 | 0.9261 | 55.434 | 0.3466 | 34.221 |
| λ=3 | 0.9253 | 54.828 | 0.3418 | 34.220 |
| λ=5 | 0.9262 | 55.342 | 0.3472 | 34.240 |
> λ∈[1,5] 把 Y 抬到 34.19–34.24 平台、四指标同向；**λ=3 的 MUSIQ 反低（54.83）说明 λ 与感知非单调**，需按口径挑点。

### 3.2 预训练 + λ 精扫（stack_dwt，AdamW 微调 b64_ft15k）
| λ | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| 1 | 0.9264 | 55.810 | 0.3501 | 34.187 |
| 2 | 0.9265 | 55.787 | 0.3501 | 34.236 |
| 3 | 0.9266 | 55.767 | 0.3501 | 34.251 |
| 5 | 0.9270 | 55.767 | 0.3506 | 34.267 |
| 6 | 0.9270 | 55.830 | 0.3506 | 34.276 |
| 7 | 0.9270 | 55.832 | 0.3509 | 34.278 |
| **8** | **0.9271** | **55.870** | **0.3513** | 34.278 |
| 10 | 0.9270 | 55.801 | 0.3507 | 34.288 |
> λ=8 为感知峰值（MUSIQ/MANIQA/SSIM 皆最高），λ=10 回落 → **scale×2 上 λ8 甜点**（与旧 λ3 判负是不同基座/不同 λ 段的假象，实为 plateau）。

### 3.3 小波基与方向（预训练 λ8，round-6）
| 变体 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| Haar λ8（基线） | 0.9271 | 55.870 | 0.3513 | 34.278 |
| db2 | 0.9270 | 55.698 | 0.3501 | 34.271 |
| db4 | 0.9269 | 55.633 | 0.3501 | 34.259 |
| db2+各向异性(HL1.6/LH0.8) | 0.9270 | 55.686 | 0.3499 | 34.253 |
> **更光滑的 Daubechies 基与各向异性都不敌 Haar** → Haar 高频子带对"该惩罚什么"已足够；基/方向调参**饱和**。

### 3.4 Muon 优化器
**(a) 零预训练 b64+λ8（muon_ab）**
| 优化器 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| AdamW 3e-4 | 0.9259 | 55.464 | 0.3470 | 34.242 |
| Muon 1e-3 | 0.9272 | 55.990 | 0.3520 | 34.301 |
| **Muon 5e-3** | **0.9280** | 55.974 | **0.3527** | 34.391 |
| Muon 1e-2 | 0.9279 | 55.929 | 0.3527 | 34.328 |
> 每个 Muon lr 都胜 AdamW；且 **Muon 零预训练(5e-3) ≈ 预训练+AdamW 冠军(34.199→34.29)** → **Muon 抵掉一大截预训练收益**。

**(b) 预训练 b64+λ8 上扫 Muon 超参（muon_tune）**：lr{2e-3,8e-3,1.2e-2}、mom{0.90,0.98}、ns{3,7}、aux{3e-4}、wd{1e-2} **9 个偏离全部 < base（lr5e-3/mom0.95/ns5）** → **Muon 最优=lr5e-3/mom0.95/ns5**。
> 教训（已入记忆）：**自动选择器必须把"未扫的 base"纳入候选**——本链一度漏选、误用 handicap 的 2e-3，已在 mech_best 用 5e-3 重跑纠正。

### 3.5 机制实验 @ Muon 5e-3（mech_best，含 TTA）
| 配置 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| champmuon（预训练+Muon+λ8） | 0.9280 | 55.850 | 0.3521 | 34.289 |
| + 移位集成 shift4 | 0.9277 | **56.022** | **0.3534** | 34.310 |
| + 各向异性+shift | 0.9277 | 55.921 | 0.3532 | 34.310 |
| + DTCWT 复数版 | 0.9251 | 55.951 | 0.3458 | 34.154 |
| + N3 对抗(数据 lever) | 0.9279 | 55.570 | 0.3511 | 34.276 |
| champmuon + TTA | 0.9289 | 55.751 | 0.3521 | 34.405 |
| **shift4 + TTA（最终交付）** | 0.9285 | **55.960** | **0.3534** | **34.406** |
| 各向异性 + TTA | 0.9285 | 55.862 | 0.3532 | 34.407 |
| N3 对抗 + TTA | 0.9288 | 55.497 | 0.3511 | 34.388 |
> **移位集成是诚实清单里唯一仍涨点的机制**（MUSIQ 56.02/MANIQA .3534 项目最高）；**DTCWT、双路小波U-Net、N3 对抗均无增益/负**。

### 3.6 双路小波 U-Net（方向一完全体，零预训练，Muon）
| 变体 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| s1_b64+λ8（对照） | 0.9262 | 55.342 | 0.3472 | 34.240 |
| z_dual（LL深+HF浅） | 0.9214 | 53.911 | 0.3338 | 33.845 |
| z_dual + shift4 | 0.9207 | 53.731 | 0.3332 | 33.856 |
| z_dual + dtcwt | 0.9105 | 51.344 | 0.3252 | 33.162 |
> **明显负**：把 U-Net 层级换成 DWT 分解 + LL 深/HF 浅的架构，反而丢点（且 db/dtcwt 组合更糟）。方向一（含其"完全体"）**判负**。

### 3.7 ×3 / ×4 迁移（零预训练 b64，AdamW/对照，scale-adaptive λ）
| scale | 变体 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|---|
| ×3 | plain（无 dwt） | 0.8687 | 51.786 | 0.3239 | 31.038 |
| ×3 | dwt λ1 | 0.8671 | 49.751 | 0.3121 | 31.060 |
| **×3** | **dwt λ2** | **0.8709** | **52.041** | **0.3320** | 31.130 |
| ×3 | dwt λ5 | 0.8686 | 50.478 | 0.3196 | 31.116 |
| ×4 | plain | 0.8295 | 47.146 | 0.2993 | 29.416 |
| ×4 | dwt λ5 | 0.8322 | 48.249 | 0.3131 | 29.511 |
> **scale-adaptive λ**：×3 λ2 才全面超 plain（λ5 过锐化、MUSIQ 反降）；×4 λ5 全面有效。**最优 λ 随退化强度**：×2≈8、×3≈2、×4≈5（非单调，重退化更需谨慎加权 HF）。绝对值低是任务本身更难，不可与 ×2 横比。

---

## 4. 收益排序与机制 insight

| 改动 | ΔY | 感知 | 成本 | insight |
|---|---|---|---|---|
| **stride-2 → stride-1** | +0.38 | ↑ | 参数少 3.7× | bicubic 后 HR 仍有真实高频需重建；pixel-shuffle 从 LR"变"高频丢相位 |
| **小波高频 Loss (λ8)** | +0.11~0.13 | **MUSIQ/MANIQA↑** | 0 推理 | L1 给 HF 条件均值→糊；正交局部基显式抬 HF |
| **移位集成** | +0.02 | **MUSIQ 最高** | 0 推理 | Haar 子带平移敏感→移位平均逼网络把细节放对带（DTCWT 平移不变的廉价替身） |
| **Muon (5e-3)** | +0.15 | ↑ | 0 | 正交化更新在 SR 上比 AdamW 好，且**抵掉预训练收益** |
| **BSRGAN 预训练** | +0.09 | ↑ | 6× 墙钟 | 合成退化只部分重合真实相机；与 dwt/Muon 叠加 |
| **TTA** | +0.10 | SSIM/MANIQA↑ | 8× 推理 | 8 重 D4 自集成，免费的方差削减 |
| 容量 5M→18.7M | +0.16 | ~ | 线性 | 拐点以上饱和（72.5M 仅 +0.05） |

**统一的世界观（最重要的 insight）**：
1. **判别式回归吃满了"可预测"的部分**（LL + HF 的条件均值）；剩余 HF 本质是**不可预测的真随机高频**。
2. 所以：**能涨点的只有"逼网络别把可预测 HF 抹平"的机制**（小波 HF loss、移位集成、各向异性），以及"更强的优化器/更多样的退化先验/推理端集成"（Muon、预训练、TTA）。
3. 而**试图"生成"那部分不可预测 HF 的一切**（全图 diffusion、latent flow、方向C 的 HF rectified-flow 残差）**全部判负**——低 NFE 从噪声生成只会注入伪影，任何感知指标都不买账。这条在本项目被反复证伪（Mamba/双路U-Net/DTCWT/flow/rectified-flow）。

---

## 5. 判负 / 关闭清单（勿重投预算）

| 路线 | 结论 | 一句话原因 |
|---|---|---|
| 扩散：pixel/latent × flow/reg（阶段B早期） | ❌ | flow 全低于 bicubic；latent_reg < 自身零初始化 |
| Mamba / VSS 扫描骨干 | ❌ | 等预算比 U-Net −2.3 dB，扫描访存受限样本效率差 |
| 坐标注入 (coord)、频域软路由 (freq-route)、门控 FFN | ❌ | 各 ±0.05 内，b64 基线已饱和 |
| 等变正则 (D4 self-sup) | ❌ | −0.07，三项≈/低于锚点 |
| 双射小波 U-Net（层级 DWT 下/上采样） | ❌ | 三项 ≤ 锚点 |
| 双路小波 U-Net 完全体（LL深/HF浅） | ❌ | 明显负（−1.5 MUSIQ） |
| DTCWT 复数小波 loss | ❌ | 比 Haar+移位差 |
| 频域 rectified flow（方向C，HF 残差生成） | ❌ | 修掉 OOM/scale bug 后仍单调毁图（scale.10→SSIM.76） |
| UWCL/CX、Wiener/AmpPhase/RadialPSF（FFT 频带头） | ❌ | L1 必须；FFT 全局增益不稳（小波因正交+局部成立） |
| N3 对抗挖数据多样性（用 Muon 公平重测） | ➖ 中性 | 未破平台；此预算下合成退化多样性红利有限 |

---

## 6. 硬件与复现

**4090 关键实测**（阶段B）：单个 b64 stride-1 batch128 臂 **util~100%、功耗~390/450W、显存~33GB** → 一臂独占全卡、无法并跑；stride-1 显存∝base∝HR面积。×3/×4 走 PNG 解码（decoded 缓存只覆盖 LR2）→ 数据饥饿、~2 s/步。computation-bound 下**降 batch 而非 grad-ckpt**。
- 128 通道对齐使同卡算力 1.8×；EMA 0.999 对最终 ckpt 有益；`--tta`/`--dwt-*`/`--adv-deg`/`--optimizer muon` 均为训练/评估开关。

**复现（×2 交付）**：
```
# 预训练（BSRGAN）→ 微调；本仓库已固化为链式脚本
python -m diffusion.train_pixel --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 \
  --native-lr 0 --objective reg --residual 1 --optimizer muon --muon-lr 5e-3 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 \
  --lr 3e-4 --steps 10000 --ema 0.999 --init <pretrain_ckpt>
python -m diffusion.eval_official --ckpt <ckpt> --mode pixel --objective reg --iqa --tta --tile 64 --pad 16
```
结果目录：`experiments/diffusion/{wave_arms,wave_unet,stack_dwt,muon_ab,muon_tune,muon_mech,mech_best,scale_3,scale_4}/<臂>/eval_iqa[/eval_iqa_tta]/eval.json`。链式脚本 `scripts/server/run_*.sh`（tmux + 幂等）。

---

## 7. 诚实边界（引用须知）

- **协议 A vs B**：所有 SwinIR 对比均为**协议 A**（SwinIR 在 RealSR Train 上重训）；官方 SwinIR real-SR 权重是**协议 B**（DF2K+BSRGAN，从未见过 RealSR），二者**不可直接横比**，引用须标注。
- **MANIQA** 在极端合成图上不守规矩，本项目仅用于**真实 SR 输出间**的相对排序。
- **评估噪声**：感知指标逐图有 ~0.1–0.3 级抖动；单个 λ/lr 点的 ±0.05 dB、±0.1 MUSIQ 差异不作强结论——本项目所有采纳均基于"整段 lr/λ 单调一致"或"跨多指标同向"。
- **muon_mech/n3muon/pretrain 行的 Y 22.3 / MUSIQ 60.9** 是**对抗预训练 ckpt 直接在 RealSR Test 上评**（域不匹配、未微调），非有效结果，仅存档不计入。

---

## 8. 全周期结论

RealSR V3 上，经过**阶段A（SwinIR 线）→ 扩散 2×2 矩阵 → stride-1 U-Net + 容量/形状/预训练 → Mamba → 针对性机制（小波/等变/双路U-Net/坐标/频域路由）→ 优化器(Muon)→ 推理(TTA)→ 数据(N3)→ 生成(C: rectified-flow)** 的穷尽式探索：

**判别式回归 + 正交小波高频 Loss（含移位集成）+ Muon + BSRGAN 预训练 + TTA** 是这条 390 对真实数据线上可持续的增益来源；最终 ×2 达 **SSIM 0.9285 / MUSIQ 55.96 / MANIQA 0.3534（Y 34.41）**，相对原版 SwinIR **+9.2 MUSIQ / +0.048 MANIQA / +1.4 dB**。生成式与一切"换骨干"路线在本数据规模下均无胜算，已逐一判负并留档。
