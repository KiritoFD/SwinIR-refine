# sr-scaling.zip — 方法清单与可移植性

来源：`third_party/sr-scaling/sr-scaling`（SISR 缩放行为研究，DF2K bicubic ×4 + RealDeg）

## 里面有什么

| 类别 | 方法 / 模块 | 路径 |
|------|-------------|------|
| **回归 SR** | SRCNN, ESPCN, EDSR/EDSR-S, SwinIR | `sisr_scaling/models.py`, `arch_swinir.py` |
| **GAN** | ESRGAN / RRDBNet + VGG 判别器 | `arch_gan.py` |
| **像素扩散** | UNet / Swin-UNet / Swin / Restormer-lite / DiT-lite | `arch_diffusion.py`, `diffusion.py` |
| **隐空间扩散** | ResShift 式 + Qwen-Image VAE / ResShift VQGAN f4 | `diffusion.py`, `vae.py` |
| **优化器** | **Muon**（Newton–Schulz 正交化）+ AdamW 辅助组 | `optim.py` |
| **退化** | bicubic；**RealDeg** 两阶段（模糊/噪声/JPEG） | `degradations.py` |
| **指标** | PSNR/SSIM/LPIPS/FID；**MUSIQ / MANIQA / CLIPIQA** | `metrics.py`, `iqa.py` |
| **数据** | DIV2K/DF2K manifest、固定 patch | `data.py` |
| **实验基建** | run_id / csv / 审计脚本（大量，可忽略） | `scripts/*`, `configs/*` |

## 他们主结论（与我们相关）

- Muon `lr=1e-3` 下 SwinIR 在 p1m→p16m 单调最好（bicubic ×4，DF2K）。  
- GAN/Diffusion 提感知与 FID，**不保证 PSNR**。  
- RealDeg = 合成两阶段退化，≠ RealSR 真实配对。

## 建议移植（按性价比）

| 优先级 | 模块 | 理由 |
|--------|------|------|
| **1 复制** | `optim.py` Muon | 他们正式回归主线；可替换/并列 AdamW |
| 2 可选 | `degradations.py` RealDeg | 若做「合成预训练 → RealSR 微调」 |
| 3 可选 | `iqa.py` pyiqa | 无参考美学指标，补报告 |
| 4 可选 | `arch_gan.py` RRDBNet | 感知对比，非 PSNR 主线 |
| 不移植 | diffusion / VAE / 巨型 scripts | 8GB + RealSR 小数据不匹配 |

## 本仓库已复制

- `model/optim.py`：Muon + `build_optimizer`（Muon 矩阵权重 + AdamW bias/norm）  
- 训练入口：`python -m model.train --optimizer muon`（见 `model/train.py`）

## 定量性能（zip 内能读到的）

| 证据 | 指标 | 数字 |
|------|------|------|
| Muon SwinIR p16m bicubic ×4 | PSNR/SSIM | **29.13 / 0.823** |
| RealDeg SwinIR p16m matched | PSNR/SSIM | **26.10 / 0.723** |
| ESRGAN p16m bicubic best | PSNR | ~27.8（低于 SwinIR） |
| ResShift official 10k backend | PSNR/LPIPS/FID | 26.99 / 0.114 / 54.9 |

**Pixel Diffusion**：文档称 formal 覆盖 30/30 train + 全套 FID/IQA，但**具体 CSV 在远端 aggregate，本 zip 无数字表**。叙事一致：diffusion 主打感知，不抢 PSNR。

## Pixel Diffusion 能否做好 RealSR？

**主观/FID 可以；配对 PSNR 很难赢回归模型。**

- 逆问题多峰：扩散是「合法解采样」，PSNR 要的是对那一张 GT 的 MSE 最优（条件均值）。
- RealSR 错位让多峰更重；406 对 + 8GB 也不适合像素扩散训练。
- 他们自己的对照：SwinIR 回归 29.13 > ResShift 官方 26.99 PSNR；ESRGAN 也低于 SwinIR PSNR。

**建议：** PSNR 主线继续 Align + ModSwinIR；扩散只在「感知/FID 另开 lane」时考虑，本轮不移植。

