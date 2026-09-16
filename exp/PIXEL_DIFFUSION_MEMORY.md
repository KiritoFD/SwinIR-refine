# Pixel Diffusion 显存需求（RealSR × DiT 参考）

**调研日：** 2026-09-16  
**参照：** `G:\GitHub\DiT`（latent DiT，ImageNet 256/512）  
**数据：** RealSR V3 ×2（本机 `data/RealSR(V3)`）  
**估算脚本：** `scripts/estimate_diffusion_vram.py`

## 0. 先分清两个「pixel」

| 术语 | 含义 | 显存 |
|------|------|------|
| **Pixel-space diffusion** | 模型直接在 RGB/HR 像素上加噪去噪 | **本节主题，贵** |
| DiT 仓库里的 **pixel mode** | 仍是 latent diffusion，只是每 step 现场 `VAE encode`（对比 latent-cached） | 只多 ~0.5G VAE |

下面说的都是 **pixel-space**。

## 1. RealSR 尺寸决定 token 数

V3 ×2 实测（抽样）：

| 集 | LR | HR (=×2) |
|----|----|----------|
| Train Canon | 均值 ~711×438，最大 1450×650 | ~1422×876，最大 ~2900 |
| Train Nikon | 均值 ~927×700，最大 1600×1075 | ~1854×1400 |
| Test | 均值 ~650–920 宽 | 全图 HR 约 1000–3000 px |

当前回归线训练用 **LR patch 64 → HR 128**。  
Pixel diffusion **不能**在全图 HR 上训；必须裁 patch。

Token 数（DiT-S，patch=2）：

\[
T=(H/p)\times(W/p)
\]

| 训练目标 HR patch | T | 相对 latent f8@256（T=256） |
|------------------|---|---------------------------|
| **128×128** | **4096** | **16×** |
| **256×256** | **16384** | **64×** |
| 512×512 | 65536 | 256× |
| latent f8, HR 128 | 64 | 0.25× |
| latent f8, HR 256（DiT 论文同级） | 256 | 1× |
| latent f4, HR 256 | 1024 | 4× |

## 2. DiT 仓库的实测锚点（latent）

`docs/training/training.md`：

| 配置 | batch | VRAM（4090 24G） |
|------|-------|------------------|
| s7 f4 latent-cached，T=256 | 224 | **19.74G** |
| s5/s6 f8 latent-cached，T=256 | 192 | 17.1G |
| 笔记本默认 | 16 | 小（脚本注释写给 laptop GPU） |

粗算：**T=256 时约 85–100MB / sample**（含激活；权重+Adam ~0.5G 另计）。

若激活近似 ∝ T（开 SDPA/FlashAttention）：

| Pixel HR patch | T | 估 batch=1 | 估 batch=4 | 8GB 卡 | 24G 卡 |
|----------------|---|------------|------------|--------|--------|
| 128 | 4096 | ~1.5G + 0.5G | **~6G** | 可跑 b2–4 | 轻松 |
| 256 | 16384 | **~6G + 0.5G** | ~22G | **b=1 勉强 / 很险** | b1–2 |
| 512 | 65536 | ~22G + 0.5G | 爆 | 不行 | b=1 也悬 |

**不开 FlashAttention** 时，注意力分数按 \(T^2\)：

| T | 单层 QKᵀ（bf16, 6 heads） | 12 层 |
|---|--------------------------|-------|
| 4096 | ~0.19G | ~2.3G |
| 16384 | **~3.0G** | **~36G → 直接不可训** |
| 65536 | ~48G | 不可能 |

结论：**HR≥256 的 pixel diffusion 必须 Flash/SDPA + 梯度检查点**；否则 24G 也不够。

## 3. 和我们 8GB / 服务器怎么对上

| 场景 | 建议 |
|------|------|
| **本机 RTX 4070 Laptop 8G** | Pixel diffusion **只考虑 HR 128**（对齐现有 LR64 crop），DiT-S，SDPA，ckpt，bf16，**batch 1–4**。HR 256 基本放弃。 |
| **服务器 ≥24G** | Pixel @ HR 256 + DiT-S/B，batch 4–16 有意义；HR 128 可上更大 batch / DiT-B。 |
| **同样卡上的 latent diffusion** | HR 256 → f8 latent T=256，和 DiT 论文同量级；24G 可 batch 192+；**8G 也能 batch 8–32**。 |
| **全图 Test（HR 1000–3000）** | 训练都用 crop；推理必须 **tile**（同 `model/eval.py`），或 latent 下采样后再解码。 |

SR 还要 **条件 LR**：若 concat 上采样 LR（3ch）→ 输入 6ch，T 不变，通道激活 ×2 量级，再吃 **~20–40%** 显存。

## 4. 和回归线对比（重要）

| 路线 | 训练显存（同 crop） | PSNR 预期 |
|------|---------------------|-----------|
| ModSwinIR-base 回归 + Align（当前 E11） | **~1.1–3G**，batch 8 | **Y 33.47**（官方全量） |
| Pixel DiT-S diffusion @128 | ~5–8G，batch 1–4 | 文献上 **通常低于** 回归；多步采样还更慢 |
| Latent DiT @256 | ~2–6G（8G 卡）/ ~20G（24G 大 batch） | 保真/纹理可能更好，**Y-PSNR 仍常弱于回归** |

`exp/DIT_LINE.md` 与 sr-scaling 审计结论不变：**PSNR 主线不要押 pixel diffusion 全量 DDPM**；更适合  
- 少步 residual / consistency，或  
- 回归式 DiT（AdaLN + RoPE），或  
- latent 生成做纹理旁路。

## 5. 一句话答案

**Pixel diffusion 在 RealSR 上的显存由 HR patch 的 token 数主导：**

- **HR 128（现配方）+ DiT-S + Flash + ckpt：约 6–8G，8G 卡可训 batch 1–4。**
- **HR 256：单卡约 6–12G（batch 1），无 Flash 直接 >24G。**
- **HR 512 / 全图：单卡 24G 也不够，必须 latent 或更小 patch/窗注意力。**

相对地，**latent f8 同分辨率只要 1/16–1/64 的 token**，24G 上可按 DiT 仓库实测跑到 batch ~200。
