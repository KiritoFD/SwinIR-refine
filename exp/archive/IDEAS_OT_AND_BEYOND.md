# 突破"配准/带宽墙"的研究创新方案（评估 + 创新一落地）

针对本项目实测的两堵墙（**配准残差锁 PSNR≈34.4–35.7dB**、**带宽截断**）与生成式 PSNR 崩塌，评估五个创新方向，并已实现**创新一（最优传输 OT）**。

## 方向评估（对齐本项目证据）

| 案 | 攻哪堵墙 | 契合度 | 成本 | 结论 |
|---|---|---|---|---|
| **① 最优传输损失 (SWD/Sinkhorn)** | 配准/点对点墙 | 极高——正打 L1 条件均值塌缩，且直接产出"MUSIQ↑/SSIM·PSNR 略降"变体 | 仅改 loss | **已实现，在跑** |
| ② 零空间扩散 (Range/Null 分解 + 物理一致性投影) | 带宽墙 | 高但风险：本项目已证生成式在 390 对崩 | 重(训扩散) | 二期候选 |
| ③ DA-INR (LIIF 隐式神经场 + 畸变位置偏置) | 离散像素/像散 | 中；任意倍率(×2.37)加分但**不碰上限** | 中 | 备选 |
| ④ TTO-LoRA (测试期自校准低秩) | 390 数据墙 | 新颖(zero-shot 内部自相似)；但逐图优化慢、易训崩 | 中 | 备选 |
| ⑤ FNO (频域算子) | 感受野/像差 | 低：**FFT 全局基本项目已多次证伪**(Wiener/AmpPhase/RadialPSF) | 高 | 不优先 |

## 创新一：OT 损失（`diffusion/ot_loss.py`，已接入 `train_pixel --ot-*`）

**机理**：L1 在**同坐标**罚像素 → 0.3px 错位即逼网络出均值(糊)。OT 把预测/ GT 的 **N×N patch 当作两个分布**比"搬运成本"，**平移不增代价** → 网络可放开生成锐利纹理，MUSIQ/MANIQA 上冲。

**两种可微实现**：
- **SWD 切片 Wasserstein**：patch 云投影到随机 1D 方向、按分位数(排序)匹配，O(M log M)、稳、默认。
- **Sinkhorn 熵正则 OT**：子采样 patch 云做 few-step 熵正则传输，O(M²·iter)，量级大需小权重。

**两种作用面**：`--ot-on image`（原始 patch）/ `--ot-on wavelet`（**Haar 高频子带**上的 OT——把"正交局部高频"与"分布匹配"结合）。可 `--ot-replace-l1` 用 OT 取代 L1 主项（保留小波/EMA）。

**在跑的 6 臂**（冠军 init + Muon + 小波λ8+shift4 为基座，A-strict 非-TTA/TTA 评估）：
`ot_swd_img_add / ot_swd_wav_add / ot_swd_wav_rep / ot_swd_img_rep / ot_sink_img_add / ot_sink_wav_add`。
对照基线 = champmuon+shift MUSIQ 55.96 / MANIQA 0.3534 / SSIM 0.9277 / Y 34.31。

**假设/判据**：若 OT 臂在 **MUSIQ/MANIQA 明显↑**（哪怕 Y 掉 0.1–0.3）→ 验证"分布匹配绕过配准墙"，并可作为交付的**感知档**（同协议可控地以保真换感知）。若 OT 臂 MUSIQ 不升或整体劣化 → 与生成式同类结论（此数据规模下分布匹配也补不回不可预测高频），如实记负。

**注意**：SWD 是 batch 内 patch 分布匹配，非严格跨图传输；权重需扫（已给 add=30/rep=8 两档，sinkhorn 用 2e-3）。结果回填 `docs/baseline` 与主表后据实取舍，不预设成功。
