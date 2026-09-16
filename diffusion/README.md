# RealSR Diffusion 实验线

**目标：** 把 diffusion 超分这条线做完整、可下结论——**latent / pixel 两条空间 × flow / reg 两种目标**的 2×2 矩阵，
全部用同一套官方 Y 指标（RealSR V3 Test 全量 100 对）对标回归线 E11（**Y 33.47 / SSIM 0.9144**）。

**参照实现：** `G:\GitHub\DiT`（AdaLN-Zero + RoPE + RMSNorm + SwiGLU + SDPA + flow matching）

## 2×2 矩阵（`--mode` × `--objective`）

| arm | 空间 | 目标 | 入口 | 想知道什么 |
|-----|------|------|------|-----------|
| **latent_flow** | Flux VAE latent (f8/16ch) | rectified flow + Heun ODE | `train_latent --objective flow` | 现代 latent diffusion SR 的真实水平 |
| **latent_reg** | 同上 | 确定性残差回归（零初始化头） | `train_latent --objective reg` | 同架构下"去掉采样"能拿回多少 |
| **pixel_flow** | RGB | flow，残差目标 `(HR−bicubic)/2` | `train_pixel --objective flow` | 经典 pixel diffusion SR 对照 |
| **pixel_reg** | RGB | 确定性残差回归 | `train_pixel --objective reg` | DiT 结构 vs SwinIR 的公平对照 |

`reg` 臂不是"另一种 diffusion"，它是**上界探针**：flow 臂的 PSNR 只能趋近条件均值
E[x0|LR]，而 `reg` 臂直接回归它。两者之差就是"采样本身"付出的 PSNR 代价。

## 关键设计（都是踩过坑才定的）

| 项 | 决定 | 原因 |
|----|------|------|
| VAE | **Flux1 VAE**（f8/16ch，Y 底噪 44.74） | SD1.5 VAE 底噪 34.6，会直接卡死 latent 路线的 PSNR |
| latent 归一化 | `(z − 0.1159) × 0.3611` | Flux latent 非零均值；不减 shift 会让 flow 的噪声→数据路径带偏置 |
| HR 目标 latent | `posterior.mode()`，**不采样** | 采样会往训练目标里注入 VAE 噪声，压低可达 PSNR 上限 |
| 训练 crop = 推理 tile | latent LR256→HR512；pixel LR64→HR128 | 训练/推理 token grid 一致，避免 RoPE 外推 |
| DiT 分辨率无关 | `grid` 由 token 数动态推导，pos_embed 恒为 0 | 原来写死 `self.grid`，带 pad 的 tile 一跑就 assert |
| 推理 tile | 重叠 2×pad + 线性交叉淡入 | 消除 tile 接缝的硬切 |
| 采样 | 固定 seed | 去掉 run-to-run 方差，PSNR 才是可比的 |
| tile batch | 一次过 `--tile-batch` 个 tile | pixel 全量评估从 ~8h 降到 ~2h |
| EMA | 0.999（不是 0.9999） | 20k step 量级下 0.9999 的窗口是 10k step，EMA 会一直落后 |

## 目录

```
diffusion/
  data.py            RealSR crop / full-pair
  vae.py             VAE presets + shift/scaling 归一化 + encode/decode(+grad)
  vae_noise_floor.py encode→decode 全图 Test 底噪
  dit.py             现代化 DiT（分辨率无关、RoPE、AdaLN-Zero、零初始化输出）
  flow.py            rectified flow + logit-normal t + Heun（可固定 seed）
  train_latent.py    latent DiT，--objective {flow,reg}
  train_pixel.py     pixel  DiT，--objective {flow,reg}
  eval_official.py   官方全量评估：tiled、交叉淡入、NFE sweep、确定性
  sample.py          对比图（LR-up / SR / HR）
  metrics.py         独立官方 Y 指标
  smoke_test.py      无 VAE 的前向/反向/零初始化/确定性 smoke
scripts/server/
  run_diffusion_matrix.sh   一键跑完整 2×2 + 评估 + 汇总表
```

## 跑法

```bash
# 服务器（4090 48G, env harness-qwen）
cd /home/ds/realsr
bash scripts/server/run_diffusion_matrix.sh          # 全矩阵
SKIP_TRAIN=1 bash scripts/server/run_diffusion_matrix.sh   # 只评估已有 ckpt
STEPS_PIXEL=12000 bash scripts/server/run_diffusion_matrix.sh

# 单条
python -m diffusion.train_latent --objective flow --vae flux1-vae \
  --lr-patch 256 --batch 32 --steps 20000 --amp --out experiments/diffusion/latent_flow
python -m diffusion.eval_official --ckpt experiments/diffusion/latent_flow/ckpt_best.pt \
  --vae flux1-vae --tile 256 --steps 20 --sweep-steps 2,4,8,16,32 --sweep-pairs 8
```

## 显存/时间（RTX 4090 48G 估计，以日志 ms/step 校正）

| arm | tokens/step | batch | 显存 | 20k step |
|-----|-------------|-------|------|----------|
| latent_flow / latent_reg | 1024 | 32 | ~8–12G | ~40–60 min（reg 更快） |
| pixel_flow | 4096 | 8 | ~10G | ~2–2.5 h |
| pixel_reg | 4096 | 8 | ~10G | ~1.5 h |
| 评估 latent（tile 256） | — | 8 tiles | ~10G | ~15 min/100 对 |
| 评估 pixel flow（tile 64, 8 步） | — | 8 tiles | ~12G | ~1.5–2 h/100 对 |

本机 8G 只能 smoke（`python -m diffusion.smoke_test`），不要在本机训练。

## VAE 选择说明

- `flux1-vae`（默认）：`diffusers/FLUX.1-vae`，f8 **16ch**，Y 底噪 **44.74**。
- `sd3-vae`：同族 16ch，可能更易拉取。
- `sd-vae-ft-ema` / `sd-vae-ft-mse`：对照（f8 4ch，底噪 34.6–35.2）。
- 本地路径：`--vae /path/to/autoencoder`（AutoencoderKL 目录），下采样/通道/scaling/shift 全部从 `config.json` 读。
- **Flux.2**：`unsloth/FLUX.2-VAE` 是单文件无 `config.json`，`AutoencoderKL` 载不了；有本地目录再 `--vae /path`。

## 底噪怎么读

`vae_noise_floor` 的 Y-PSNR 是 latent 路线的**硬上限**。Flux 44.74 ≫ E11 33.47，
所以 latent 路线的瓶颈**不是 VAE**，而是：数据/步数预算 + 采样目标（flow vs reg）+ 条件注入方式。

## 详细实验设计与判读规则

见 `exp/DIFFUSION_PLAN.md`。
