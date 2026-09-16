# RealSR Diffusion 实验线

**状态：** 代码就绪，等服务器跑（不在本机 8G 上训 pixel HR256 / 大 batch）  
**参照：** `G:\GitHub\DiT`（AdaLN-Zero + RoPE + RMSNorm + SwiGLU + flow matching）

## 计划（与用户约定）

| 步 | 内容 | 入口 |
|----|------|------|
| 1 | **VAE 重建底噪**（选最新 VAE，如 Flux/SD3 16ch） | `python -m diffusion.vae_noise_floor` |
| 2 | **Latent DiT** SR（LR-up latent 作条件） | `python -m diffusion.train_latent` |
| 3 | **Pixel DiT** SR（bicubic-up LR concat） | `python -m diffusion.train_pixel` |
| 4 | 官方 Y 指标对比 vs E11 回归线 33.47 | `model/eval` 或后续 sample 脚本 |

## 目录

```
diffusion/
  data.py            RealSR crop / full-pair
  vae.py             preset: sd-vae-ft-ema, flux1-dev, flux1-schnell, sd3-vae, 本地路径
  vae_noise_floor.py encode→decode 全图 Test 底噪
  dit.py             现代化 DiT（自包含）
  flow.py            rectified flow + logit-normal + Heun
  train_latent.py    latent DiT
  train_pixel.py     pixel DiT
  smoke_test.py      无 VAE 的前向/反向 smoke
scripts/server/run_pipeline.sh
```

## 显存预算（见 `exp/PIXEL_DIFFUSION_MEMORY.md`）

| 配置 | 约需 | 机器 |
|------|------|------|
| VAE noise floor tile=256 | 2–4G | 本机/服务器 |
| Latent DiT-S @32², batch 16 | ~6–10G | 本机可 smoke，正式上服务器 |
| Pixel DiT-S @128, batch 4 + ckpt | ~6–8G | 8G 可 smoke |
| Pixel DiT-S @256, batch 4 | ~22G | **仅 24G 服务器** |

## 服务器

```bash
# 本机推送（需 ds@10.222.120.101 已加入公钥）
ssh ds@10.222.120.101 'mkdir -p ~/realsr'
scp -r diffusion scripts/server model G:\RealSR\diffusion\* ds@10.222.120.101:~/realsr/   # 见 sync 脚本

# 在服务器
export ROOT=$HOME/realsr
# 数据：把 RealSR(V3) 放到 $ROOT/data/RealSR(V3)
SMOKE=1 bash scripts/server/run_pipeline.sh flux1-dev
# 正式
STEPS=20000 bash scripts/server/run_pipeline.sh flux1-dev
```

## VAE 选择说明

- `flux1-dev` / `flux1-schnell`：FLUX.1 系 VAE，f8 **16ch**（与 SD3 同族），更新、重建通常优于 SD1.5 f8 4ch。
- `sd3-vae`：同族 16ch，有时更易拉取。
- `sd-vae-ft-ema`：对照（f8 4ch）。
- 本地路径：`--vae /path/to/autoencoder`（AutoencoderKL 目录）。

**Flux.2：** 若服务器已有权重目录，直接 `--vae /path/to/flux2/vae`；代码按 `AutoencoderKL` 加载，不依赖写死的 HF id。

## 解读底噪

`vae_noise_floor` 的 **Y-PSNR 是 latent 路线的上限**：decode 误差无法被生成模型补回。  
若 Flux VAE 底噪 Y≈38，则 DiT 采样 Y 不应期望超过 ~38；回归 SwinIR E11=33.47 仍可能赢 PSNR。

## 本地

```powershell
python -m diffusion.smoke_test
```
