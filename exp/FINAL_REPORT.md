# RealSR V3 ×2/×3/×4 — 最终整合报告（2026-09-10 → 09-24）

> 单页全貌。任务/协议/地板、冠军配方、有效项排名、已判负路线（含本轮 Mamba/等变/双射小波U-Net/频域rectified flow）、方法论、开放项。
> 细节见 `STAGE_SUMMARY.md`、`WAVE_ARMS.md`、`MAMBA_MATRIX_RESULTS.md`、`NEW_ARMS.md`、`RESULTS_SUMMARY.md`。
> **评价口径（用户定）：PSNR-Y 次要，主看 SSIM / MUSIQ / MANIQA。** 数字为官方 100 对 Test.m，limited-range Y/uint8，全仓库同一 `eval_official`。

## 0. 一页结论

**交付冠军（×2）：`s1_b64` + BSRGAN 预训练 + 小波高频 Loss(λ8) + EMA + TTA**
**SSIM 0.9279 · MUSIQ 55.83 · MANIQA 0.3516 · (Y 34.38)**

| 基线 | SSIM | MUSIQ | MANIQA | (Y) |
|---|---|---|---|---|
| bicubic 地板 | 0.8876 | 39.40 | 0.2782 | (31.73) |
| SwinIR-largeish（原版重训） | 0.9058 | 46.79 | 0.3051 | (32.97) |
| E11 Mod-SwinIR（阶段A旧改进线） | 0.9144 | 49.15 | 0.3089 | (33.47) |
| **交付冠军** | **0.9279** | **55.83** | **0.3516** | (34.38) |
| **Δ vs SwinIR-largeish** | **+0.022** | **+9.0** | **+0.047** | (+1.40 dB) |

一句话：**相对原版 SwinIR 是一个跨量级的感知提升**（MUSIQ +9 / MANIQA +0.047），且 PSNR 也 +1.4 dB。

## 1. 任务与协议
RealSR V3 真实相机配对，Train 406 对（16 早停 val，实训 390），Test 100 对（Canon50+Nikon50），×2/×3/×4。对齐官方 `Test.m`：RGB 训练 / **Y 测试**、limited-range BT.601、uint8、modcrop4、不 shave；`eval_official` 一套代码 + MUSIQ/MANIQA 逐图。协议 A（都见过 RealSR train）才可与 SwinIR 重训横比；官方 SwinIR real-SR 是协议 B（DF2K+BSRGAN，没见过 RealSR），引用须分清。

## 2. 冠军配方（stride-1 U-Net + 小波）
`--backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 --attn-levels 2,3 --native-lr 0`
- **残差 bicubic + 全零初始化头** → step0 精确 = bicubic 地板（免费诊断不变量）。
- 主损失 L1 + **正交 Haar 小波高频子带(HL/LH/HH)多尺度 L1，λ=8**（`diffusion/wavelet.py`，零推理成本）。
- **scale-adaptive λ**：×2 λ8、×3 λ2、×4 λ5（退化越重 λ 越小，否则过锐化伤感知）。
- **BSRGAN 预训练 → RealSR 微调**；EMA 0.999；lr 3e-4 cosine、batch 128、10000 步；bf16 AMP。
- **TTA**（推理 8 重 D4 自集成，`eval_official --tta`）：SSIM/MANIQA/Y 再抬、MUSIQ 持平。
- 参数量 18.67M；全流程 4090 单卡 ~6h（预训练 4.8h + 微调）。

## 3. 有效项（收益排序，架构 ≫ 容量 > 数据 > 步数）
1. **stride-2 → stride-1（目标分辨率运算）：+0.38 dB、参数少 3.7×**。全项目最大单项。
2. **小波高频 Loss：+0.11~0.13 dB 且感知同升**（λ 平台 34.19→34.28）——**本大波唯一重大增益的机制**。
3. **TTA 自集成：SSIM/MANIQA/Y 免费涨**（冠军即含）。
4. 容量 5M→18.7M +0.16（拐点以上饱和，72.5M 仅 +0.05）；128 通道对齐 → 同卡算力 1.8×。
5. BSRGAN 预训练 +0.09；offset-aligned L1、残差-bicubic 零初始化头（基础设施级）。

## 4. 已判负路线（别再开；本轮新增 4 条）
| 路线 | 结论 | 关键证据 |
|---|---|---|
| **扩散/flow（全图）** | ❌ 此预算不及回归 | 所有 flow 臂 < bicubic；latent_reg 甚至 < 自身零初始化 |
| **Mamba / VSS 扫描** | ❌ 关线 | 等预算(328k) U-Net +2.3 dB；Mamba 4.7× 预算才追平 |
| 坐标注入 / 频域软路由 / 门控 FFN | ❌ ≈打平/微负 | 各 ±0.05 内，b64 基线饱和 |
| **等变正则(D4)** | ❌ | Y −0.07，三项 ≈/低于锚点 |
| **双射小波 U-Net(方向一)** | ❌ | 重评后三项 ≤ 锚点；无损 stride-2 不涨点 |
| **频域 rectified flow(方向C, HF残差)** | ❌ | 修掉 attn-OOM+scale bug 后仍单调毁图(scale.1→SSIM.76)；不可预测高频≠可生成 |
| UWCL/CX 主损失、Wiener/AmpPhase/RadialPSF(FFT 频带头) | ❌ 负收益 | L1 必须；FFT 全局增益不稳（小波因正交+局部成立） |

## 5. 方法论收获
1. **先建地板再谈超越**；**评估协议是结论的一部分**（同模型子集/口径差 0.5 dB）；定生死用官方 100 对 + IQA，±0.05 算平局。
2. **感知指标必须与保真一起看**：distortion–perception 分离（flow PSNR 低但 MUSIQ 高）。
3. **等预算对照**（Mamba vs U-Net 的样本预算对齐）才归因得准。
4. **换骨干/堆参数/加数据/加长普遍撞墙**时，小波高频 Loss 这种"针对性正交机制"才涨点——但**生成式在 390 对规模下无胜算**（判别式回归吃满可预测部分，不可预测高频生成只会注入伪影）。
5. computation-bound：显存不足**降 batch，不用 grad-ckpt**。

## 6. 开放项 / Muon 结论
- **Muon 优化器：判为有效增益（本轮 09-24 结论）**。zero-pretrain b64+dwtλ8 上配对扫 lr（只换优化器），**每个 lr 点 SSIM/MUSIQ/MANIQA 全超 AdamW**（非单点侥幸）：
  | 优化器 | SSIM | MUSIQ | MANIQA | (Y) |
  |---|---|---|---|---|
  | AdamW 3e-4 | 0.92590 | 55.46 | 0.3470 | (34.242) |
  | **Muon 5e-3（甜点）** | **0.92802** | 55.97 | **0.3527** | (34.391) |
  （1e-3/2e-3/1e-2 均介于其中，MUSIQ 55.8–56.0）。“零预训练+Muon 5e-3”已超“预训练+AdamW”的前冠军 b64_pre_dwt8(.9271/55.87/.3513) → **Muon 抵一大截预训练收益**。
- **正在跑**：`b64_pre_dwt8_muon` = 预训练 init + Muon 5e-3 + λ8 + TTA（交付态叠加优化器），已完成：**SSIM .92797 / MUSIQ 55.85 / MANIQA .3521 / Y 34.289（+TTA .92890/55.75/.3521/34.405）**。
- **Muon 参数扫点（Phase M）**：在交付基座上扫 lr/momentum/ns_steps/aux-lr/wd，**9 个偏离全部不敌 base lr=5e-3** → **Muon 最优=lr5e-3/mom0.95/ns5**（选择器曾漏选 base、已修）。
- **mech@5e-3（诚实清单机制项）**：**移位小波 loss p_shift4 = 新非-TTA最优**（**MUSIQ 56.02 / MANIQA .3534 全项目最高**、SSIM持平、Y34.31）；各向异性≈持平；**DTCWT 更差**（死）；**双路小波U-Net 明显负**（死）。**新交付候选 = 预训练+Muon5e-3+λ8+shift4**（+TTA 链式评估中）。N3 对抗(Muon) pretrain→ft 待收。

> 诚实清单总结：四条非回归路线（Mamba/等变/双射小波U-Net/频域rectified-flow）+ 双路小波U-Net + DTCWT 均判负；**只有“小波高频 Loss”及其“移位集成”持续涨点**（已与预训练/Muon/TTA 叠加）。
- N3 对抗数据多样性（唯一未验的数据 lever，现已在 mech_best 里用 Muon 跑，待收）。

> **48h 预算重排（硬件实测）**：单个 b64 stride-1 臂已把 4090 跑满（util~100%、390/450W、33/48GB）→ 一个臂独占 ~1.7–3.3h，无法并跑，故 20 个配置扫点=实打实~48h、但边际收益低。已改投**诚实清单**（`run_honest48.sh`）：N3 对抗挖数据多样性（G_phi 攻含小波-HF 的目标）+ Muon momentum/ns_steps + 每尺度独立 λ + LL 项。新增 flag：`--muon-momentum/--muon-ns-steps/--dwt-level-weights`。

## 7. 产物
`experiments/diffusion/{wave_arms,wave_unet,stack_dwt,scale_3,scale_4,muon_ab}/<臂>/eval_iqa[/eval_iqa_tta]/eval.json`；脚本见 `scripts/server/run_*.sh`（链式/幂等）。
