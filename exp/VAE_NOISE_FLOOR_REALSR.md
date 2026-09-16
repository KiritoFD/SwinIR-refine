# VAE 重建底噪结果（RealSR Test，服务器 dserver 4090）

**日期：** 2026-09-16  
**数据：** RealSR V3 ×2 Test（n=15 抽样，含 Canon/Nikon，全图 HR）  
**代码：** `diffusion/vae_noise_floor.py`  
**服务器：** `ds@10.222.120.101` · RTX 4090 48G · env `harness-qwen`  
**结果文件：** `/home/ds/realsr/experiments/diffusion/vae_noise/`

## 汇总（encode→decode 纯重建，无扩散）

| VAE | 下采样 | latent ch | **Y-PSNR** | Y-SSIM | RGB-PSNR | 来源 |
|-----|--------|-----------|------------|--------|----------|------|
| **flux1-vae** | f8 | **16** | **44.74** | **0.9920** | **42.55** | `diffusers/FLUX.1-vae`（公开镜像） |
| sdxl-vae | f8 | 4 | 36.06 | 0.9482 | 34.26 | `madebyollin/sdxl-vae-fp16-fix` |
| sd-vae-ft-mse | f8 | 4 | 35.16 | 0.9396 | 33.44 | `stabilityai/sd-vae-ft-mse` |
| sd-vae-ft-ema | f8 | 4 | 34.62 | 0.9346 | 32.91 | `stabilityai/sd-vae-ft-ema` |

## 结论（步骤 1）

1. **选型：latent 路线必须用 Flux1 VAE（16ch）**  
   - 比 SD1.5 f8 高 **~10 dB Y**，比 SDXL VAE 高 **~8.7 dB**。  
   - 重建上限 44.7 dB ≫ 回归 SOTA E11 的 33.47 → **VAE 不再是 PSNR 瓶颈**，瓶颈会在生成模型/采样步数。

2. **对照回归线**  
   - E11 回归 Y=33.47（官方全量）  
   - 即使用最差 VAE（sd-vae-ft-ema），上限 34.6 也刚够到 E11；**SD1.5 VAE 会直接卡死 latent 扩散的 PSNR**。

3. **Flux.2 VAE**  
   - `unsloth/FLUX.2-VAE` 为单文件格式（无 `config.json`），`AutoencoderKL.from_pretrained` 不能直接载。  
   - 当前用 **Flux.1 公开 VAE**（同族 16ch f8）作为「最新可用」；若服务器有本地 Flux.2 目录可 `--vae /path`。

4. **网络**  
   - 服务器访问 `huggingface.co` 不可达；`HF_ENDPOINT=https://hf-mirror.com` 可用。  
   - BFL 官方 `FLUX.1-dev` 仓库在镜像上不可见；用 `diffusers/FLUX.1-vae` 替代。

## 下一步

- [ ] Train 数据上传后：latent DiT（`--vae flux1-vae`）  
- [ ] pixel DiT HR128  
- [ ] official full-set eval vs E11  

```bash
export HF_ENDPOINT=https://hf-mirror.com
export PY=/home/ds/miniconda3/envs/harness-qwen/bin/python
cd /home/ds/realsr
$PY -m diffusion.train_latent --vae flux1-vae --batch 32 --amp --steps 20000
```
