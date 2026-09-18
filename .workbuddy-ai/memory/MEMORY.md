# RealSR 项目长期记忆

## 硬件：RTX 4090 48G（`ds@10.222.120.101`，env `harness-qwen`）

**选 batch 按吞吐，不按显存占用。** 这张卡超过 ~40G 后呈超线性变慢，把显存"占满"反而降吞吐：

- latent-S：b=64 (23.8G, 132 smp/s) 优于 b=112 (41.0G, 125 smp/s)
- latent-XS：b=96 (12.0G, 328 smp/s) 远优于 b=320 (39.5G, 210 smp/s)
- pixel-S：b=28 (40.8G, 29.6 smp/s) 略优于 b=24 (35.1G, 28.0 smp/s)
- pixel-XS：b=32 (15.7G, 62.7 smp/s) 优于 b=64 (31.2G, 56.0 smp/s)

未 compile 的小 DiT 约 30% MFU（~25-30 TFLOPS），属正常。要真正提速得上
`torch.compile` / kernel fusion，不是调 batch。box 有 24 核，DataLoader 默认
4 workers 偏少（pixel 臂给到 8-16）。

## 环境

- `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 不通）
- 数据：Train 406 对，Test 100 对；3 个 VAE 的 latent 缓存已建在 `data/latents/<vae>/`
- 本机 Python 有 torch 的是 `C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe`（只够 smoke）

## 约定

- 服务器上改 `.sh` 后确认是 LF；Windows 侧用 Python `write_text` 会写出 CRLF，推上去会 `syntax error`
- 正在运行的 `run_36h*.sh` **不要就地编辑**（bash 按字节偏移续读会执行乱码）
- 官方指标：limited-range Y (BT.601 16-235)、uint8、crop=0、shave=0、100 对均值
- 参考锚点 E11：ModSwinIR-base + Align-L1，4.05M，12k 步，**Y 33.47 / SSIM 0.9144**
- **地板（100 对官方协议，必背）**：bicubic **31.7335**；bicubic 过一遍 VAE 往返
  **31.5971**（= reg 臂零初始化时的输出，因为输出头零初始化）。任何 latent/flow 臂
  如果低于 31.60，就等于「连什么都不做都不如」。

## torch.compile 实测

- latent-S b=64：0.485 → 0.384 s/step（**1.26×**），值得开
- pixel-S b=28：0.947 → ~0.90 s/step（1.05×），几乎没加速，但**显存 40.8G → 20.7G 减半**
- 结论：token 数多、MLP 重的臂（latent）吃 compile 收益；小 patch attention 的臂基本只省显存

## 排障

- **不要 `pkill -f "<脚本名>"`**：ssh 远端那条 `bash -c ...` 的命令行里就含这个串，
  会把自己一起杀掉 → exit 255 + 空输出。用 `[c]hain_afte` 这类自排除模式。
- `start_36h.sh` 遇到同名 tmux session 会打印提示并 `exit 0`（不是报错）。
  被 kill 的链路常留下**死 session**，会让后续阶段静默跳过 —— 启动前先 `tmux kill-session`。
- 想给 tmux 新 session 传变量，别指望 `ENV=x start_36h.sh`；直接写进
  `tmux new-session -d -s X "export A=1 && bash ..."`。
- 后台链路要用带锁的启动器（如 `start_chain2_once.sh` 的 mkdir 原子锁）。
  单靠 `pkill` 清理极易留下 0 个或 2 个实例；两个实例会互相 `kill-session`。

## 当前最强：stride-1 U-Net b64（`--backbone unet --base 64 --mult 1,2,4,4 --native-lr 0`）

官方 100 对 **Y 34.1083 / SSIM 0.9246**，**零预训练**，纯 RealSR，18.67M 参数，batch 128，10000 步。

| 参考 | 参数 | Y |
|---|---|---|
| **U-Net b64 stride1** | 18.67M | **34.108** |
| U-Net b40 stride1 | 7.58M | 34.007 |
| U-Net b32 stride1 | 4.99M | 33.945 |
| DiT pixel_reg | 34.0M | 33.763 |
| E11（ModSwinIR+Align） | 4.05M | 33.47 |
| Stock SwinIR-largeish | ~10M | 32.97 |
| bicubic 地板 | — | 31.73 |

**容量拐点在 10M 以上**：4.99M→10.73M 完全平坦（33.94/34.01/33.99），18.67M 才 +0.10。

**`--native-lr 0`（在 HR 尺度运算）是关键**：同为零预训练，stride-2 的 18.68M U-Net 只有
33.56，而 stride-1 的 4.99M 就有 33.94。**在目标分辨率上运算比参数量重要得多。**
代价是显存 ∝ base（b32 138 / b40 172 / b48 207 / b64 276 MB/样本）和硬件效率下降
（C=32 时 GEMM 瘦，只跑到 4.4 TFLOPS vs 10.7）。

**容量 5M~10.7M 之间无差别**（33.94/34.01/33.99）—— 容量不是瓶颈，这一点在 DiT 上也验证过。

**预训练 batch 必须按尺寸调**：`PRE_BATCH` 固定 256 会让 b48/b64 直接 OOM。

## 骨干：DiT vs U-Net（`diffusion/dit.py` / `diffusion/unet.py`）

两者接口完全一致（`forward(x,t)`，x=(B,C,H,W)，t∈[0,1000]），`--backbone {dit,unet}` 切换。

| 模型 | 参数 | batch | 显存 | samples/s @HR128 |
|---|---|---|---|---|
| DiT-S | 34.0M | 28 | 20.7G | 70 |
| **U-Net base=64** | 18.7M | 390 | 21.7G | **940** |

**U-Net 的 native-LR 是关键**（`in_stride=2, out_scale=2`）：stem 一个 stride-2 卷积降到
LR 尺度，编码器/解码器全在 LR 尺度跑（skip 落在相同分辨率层上），最后 pixel-shuffle 抬回 HR。
最高分辨率上只剩一个 stem 卷积和一个 shuffle 卷积 → 省 29% FLOPs、**显存降 4.5 倍**。
⚠️ 只对 reg 有效；flow 的 ODE 状态在 HR 带白噪声，下采样会混叠掉速度场要重建的信息。

`model.align = in_stride × 2^(levels-1) × out_scale`（=32）。eval 的 tile 必须按它对齐，
且图片尺寸不是 align 整数倍时要先 pad 再裁回（RealSR 的 1000%32=8，不处理会 skip concat 崩）。

**DiT-S 的算力去向**：HR128 → n=4096 tokens，**attention 的 n² 项占 64% 的 FLOPs**，
整步 101 TFLOPS = 4090 峰值的 61%（真算力瓶颈）。但存算比已 1962 FLOP/byte
远超 ridge point 165 → **提高存算比无收益**，真正的杠杆是减少全分辨率 attention。

## 数据管线

`data/decoded/`：4562 张图 / 31.23 GB 预解码 uint8 blob（`precache_hr.py`，21 秒建好）。
`DecodedStore` 按**文件路径**索引，任何 dataset 直接 `store.get(path)`，零拷贝 memmap。
- RealSR 10.95 ms/item（解码瓶颈已消除）
- **BSRGAN 153 ms/item —— 瓶颈是退化本身（JPEG 往返），不是解码**；12 worker 只有 84 items/s，
  而 batch 390 @0.42s 需要 929 samples/s。预训练前必须先解决这个。

## 评估成本（排程前必算）

像素空间 flow 的官方评估比 latent 贵一个数量级：实测 **9–10 s / NFE / 对**
（1000×1400 图，tile 64 → HR core 128，约 342 tile/图，34M DiT）。
- pixel flow 100 对 @NFE16 实测 **~4 h**（sweep 另计）；NFE64 ≈ 10.5 h，跑不起。
  一次带 sweep(2,4,8,16,32)+官方 100 对 @NFE16 的完整评估实测 **5.2 h**
- latent flow 100 对 @NFE40 只要 ~7 min（tile 256，8× 下采样，token 少 64 倍）
- reg 臂 100 对 ~12 min（单遍）
→ 像素 flow 只能报低 NFE 官方数 + 子集 NFE 饱和曲线。

`eval_official.py` 经 `| tee` 时 stdout 块缓冲，日志几十分钟不刷新；
**别用 tail 判进度**，看进程存活 + `nvidia-smi`。
