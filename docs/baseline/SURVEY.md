# 五年顶会超分 Baseline 全景 & 复现清单（2021–2026）

检索范围：CVPR / ICCV / ECCV / NeurIPS / ICLR / SIGGRAPH + TPAMI / IJCV，任务聚焦**图像单帧超分**中与 **RealSR 真实世界评测**相关者。
结论先给：**五年内顶会超分方法总数 > 40**；但"能/该在我们这条 RealSR×2 配对保真+感知赛道上比的"是其中一部分，**需真正复现的只有约 8 个（判别式架构基线）**，其余 **约 25 个引用文献数字（带协议标签）**。

---

## 1. 全景分类与计数

### A. 架构基线（经典/合成退化 SOTA，追 PSNR）— 2021+ 约 12 个
| 方法 | 出处/年 | 备注 |
|---|---|---|
| SwinIR | ICCV-W 2021 | 已复现（light/largeish） |
| ELAN | ECCV 2022 | Transformer 效率 |
| HAT | CVPR 2023 | 现代主基准（宽通道） |
| DAT | ICCV 2023 | 双聚合 Transformer |
| SRFormer | ICCV 2023 | 置换自注意力 |
| RGT | NeurIPS 2023 | 视网膜-皮层 Transformer |
| CAT | ICCV 2023 | 通道感知 Transformer |
| OPSA | CVPR 2023 | 偏置自注意力 |
| SANL / NAFNet-style | CVPR 2022 | 无归一化 |
| ATD | CVPR 2024 | 自适应 token 字典 |
| MambaIR | ECCV 2024 | Mamba 复原（我们 Mamba 线相关） |
| RLFN | IJCV 2023 | 残差稀疏（高效赛道基线） |
（2021 前经典仍被当基线：**SRCNN/VDSR/EDSR/RCAN/RDN**——已/在复现。）

### B. 真实世界 / 盲超分（退化建模 + GAN）— 约 12 个
BSRGAN(ICCV21) · Real-ESRGAN(ICCVW21) · KDSR(ICCV21) · DASR(ECCV22) · LDL(CVPR22) · FeMaSR(CVPR22) · MM-RealSR(CVPR22) · SPSR(CVPR23) · ArbSR(CVPR23) · CLPARB(CVPR22) · DR2Net/Adaptive-Dropout(CVPR25) · RealSR-R1(2025, RL)

### C. 扩散 / 一步扩散真实超分 — 约 16 个
DiffBIR(ICCV23) · StableSR(IJCV24) · ResShift(NeurIPS23) · SeeSR(CVPR24) · PASD(ECCV24) · SUPIR(CVPR24) · OSEDiff(NeurIPS24) · SinSR(CVPR24) · S3Diff(CVPR24) · AddSR(2024) · AdcSR(CVPR25) · TSD-SR(CVPR25) · PiSA-SR(CVPR25) · InvSR(ICLR25) · RCOD(2025) · TinySR(2025) · PURE(2024) · RealSR-R1(2025)

**合计 ≈ 40+（2021–2026）**；含经典架构则 ≈ 45+。

---

## 2. 需要复现 vs 只引用（判据 = ①是否在 RealSR 配对协议下可比 ②能否用我们一台 4090、无外部权重/无外网下载 复现）

| 处置 | 数量 | 方法 | 理由 |
|---|---|---|---|
| **复现（A-strict 同协议自训）** | **~8** | SRCNN ✓ · VDSR ✓ · EDSR ✓ · RCAN(跑) · RDN(跑) · SRResNet(跑) · RRDB/Real-ESRGAN-body(跑) · SwinIR ✓（**建议补 HAT / DAT**） | 判别式架构，可从 RealSR-Train 从零训、同评测器出 PSNR/SSIM/MUSIQ/MANIQA——唯一"公平横比"的那张表 |
| **引用 + 协议标签** | ~12 (B) | BSRGAN·Real-ESRGAN·KDSR·LDL·DASR·FeMaSR·SPSR·ArbSR·MM-RealSR·CLPARB·DR2Net·RealSR-R1 | 多为 GAN/需其训练配方；报的是 ×4/剃边或合成退化输入 → 标协议引用；其官方**权重需外网下载**（服务器不可达）|
| **引用 + 协议标签** | ~18 (C) | StableSR·DiffBIR·ResShift·SeeSR·PASD·SUPIR·OSEDiff·SinSR·S3Diff·AdcSR·TSD-SR·PiSA-SR·InvSR·RCOD·TinySR·AddSR·PURE | 依赖 **Stable-Diffusion 基座 + 大训练集(LSDIR/FFHQ)**，一台 4090 + 无外网无法忠实复现；引 TinySR/RealSR-R1 同协议表 |
| **不纳入** | 若干 | 任意倍率/视频/参考图/高光谱/遥感 SR | 非本任务 |

**净结论**：**真正需要复现的只有 A 类判别式架构 ~8 个（已含全部在跑）**；B/C 两类共 ~30 个方法**用文献数字 + 明确协议标签**呈现（RealSR×4 那批来自 TinySR 2508.17434 / RealSR-R1 2506.16796 的统一表）。

---

## 3. 缺口 / 建议补的复现

当前复现流水线：SwinIR、EDSR、RCAN、SRResNet、RRDB、SRCNN、VDSR、RDN（8 个，架构全谱覆盖 CNN→注意力→RRDB）。
若要把 A 类做满（审稿人常点名）：**再补 HAT 与 DAT** 两个现代 Transformer 主基准（从零训，A-strict 评测）。二者实现较重（HAB / 双聚合块），单个训练 ~2–3h；如要，我下一步加 `--backbone hat|dat` 并入链。
可选：加 **NTIRE2025 ×4 冠军**类经典方法，但其在 RealSR 配对保真非主赛道，优先级低。

---

## 4. 与感知调优的关系（本波同时在做）
除 baseline 外，另跑 **6 个感知变体**（用缓存 MUSIQ/MANIQA 作可微损失 + 高 λ 锐化），产出"MUSIQ/MANIQA 高、SSIM/PSNR 略降"的一族——对应 B/C 类"以保真换感知"但**在我们的 A-strict 配对协议**下的可控版本。结果表将单列。
