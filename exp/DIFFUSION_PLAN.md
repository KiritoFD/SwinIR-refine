# Diffusion 超分路线：如何做完整（latent + pixel 两条）

**日期：** 2026-09-16
**机器：** `ds@10.222.120.101` · RTX 4090 48G · env `harness-qwen` (torch 2.10+cu128)
**对标：** 回归线 E11（ModSwinIR + Align-L1，12k）**Y 33.47 / SSIM 0.9144**（全量 100 对）
**已完成：** 步骤 1 VAE 底噪 → Flux1 VAE **44.74 dB Y**（文档 `exp/VAE_NOISE_FLOOR_REALSR.md`）

---

## 1. 现在的位置

| 已完成 | 内容 |
|--------|------|
| ✅ | VAE 选型：Flux1 VAE f8/16ch，Y 底噪 44.74 ≫ E11 33.47 → **VAE 不再是瓶颈** |
| ✅ | 现代化 DiT（AdaLN-Zero + RoPE + RMSNorm + SwiGLU + SDPA）+ flow matching |
| ✅ | 官方全量评估脚本（Y/SSIM，与 `model/eval` 同协议） |
| ⏳ | Train 数据上传（Canon→Nikon，tarball 2.3G/3.7G） |

**关键判断：** latent 路线的上限 44.7 dB 远高于回归线，所以**差的不是 VAE，是"怎么学"**。
接下来唯一要回答的问题是：**同样的 DiT、同样的数据、同样预算下，扩散采样到底比确定性回归多/少多少 dB。**

---

## 2. 本轮代码改动（都已本地 smoke 通过）

踩到的坑，都会直接毁掉结果，所以先修再跑：

| # | 问题 | 修法 |
|---|------|------|
| 1 | **DiT 写死训练分辨率**：`forward` 用 `self.grid`，带 context pad 的推理 tile 直接 `assert` 崩 | `grid` 由 token 数动态推导；`pos_embed` 恒 0（RoPE 提供位置），模型分辨率无关 |
| 2 | **Flux latent 没减 shift**：只做了 `z*0.3611`，latent 均值 ~0.32，flow 的噪声→数据路径带偏置 | `encode = (z − 0.1159)*0.3611`，`decode` 取逆；`scaling/shift/channels/downscale` 一律从 `config.json` 读 |
| 3 | **HR 目标 latent 用 `posterior.sample()`**（默认 `--sample-hr 1`），往训练目标里注入 VAE 噪声 | 默认改 `mode()`，且与条件 latent 一致 |
| 4 | **训练 crop ≠ 推理 tile**：训 HR128/latent16，评估用 HR256/latent32 | latent 默认 LR256→HR512；pixel 保持 LR64→HR128（与回归线一致）；`--tile` 自动取训练 crop |
| 5 | **tile 硬切**：相邻 tile 无重叠，接缝处上下文不连续 | 每个 tile 带 `pad` 上下文，重叠 2×pad，线性交叉淡入（已验证与 torch bicubic 参考 maxdiff=0） |
| 6 | **采样无 seed**：每次评估初始噪声不同，PSNR 不可复现 | `sample_flow(..., seed=)`，CUDA generator |
| 7 | **EMA 0.9999** 对 20k step 太慢（窗口 10k step，EMA 一直落后） | 默认 0.999 |
| 8 | **pixel 全量评估 ~8h**（逐 tile batch=1） | tile 按形状分组批量推理 `--tile-batch 8`，OOM 自动减半 → ~1.5–2h |
| 9 | **只有 flow 一条路，没法回答"采样值多少"** | 新增 `--objective {flow,reg}`（见下） |

---

## 3. 实验设计：2×2 矩阵

| arm | 空间 | 目标 | 入口 | 回答什么 |
|-----|------|------|------|----------|
| **latent_flow** | Flux latent | rectified flow + Heun | `train_latent --objective flow` | 现代 latent diffusion SR 的真实水平 |
| **latent_reg** | Flux latent | 确定性残差回归（零初始化输出头） | `train_latent --objective reg` | 同架构"去掉采样"能拿回多少 dB |
| **pixel_flow** | RGB | flow，残差目标 `(HR−bicubic)/2` | `train_pixel --objective flow` | 经典 pixel diffusion SR 对照 |
| **pixel_reg** | RGB | 确定性残差回归 | `train_pixel --objective reg` | DiT 结构 vs SwinIR 的公平对照 |

**`reg` 臂不是"另一种 diffusion"，它是上界探针。**
flow 的 PSNR 只能趋近条件均值 E[x₀|LR]（要很多 NFE 才逼近，且带采样方差）；`reg` 直接回归它。
`gap = PSNR(reg) − PSNR(flow@NFE)` 就是**采样本身付出的 PSNR 代价**——这是整条线唯一真正想要的结论。

四个 arm 共用：DiT-S、同一 RealSR V3 ×2 406 对、同一 crop、同一 EMA/cosine schedule、同一官方评估。

### 配置

| arm | LR crop | HR | tokens | batch | steps | 显存 |
|-----|---------|----|--------|-------|-------|------|
| latent_flow / latent_reg | 256 | 512 (latent 64²) | 1024 | 32 | 20k | ~8–12G |
| pixel_flow / pixel_reg | 64 | 128 | 4096 | 8 | 20k | ~10G |

### 评估协议

- 全量 **100 对** Test（Canon+Nikon），tiled，Y-PSNR / Y-SSIM，shave=0 —— **与 E11 完全同协议，可横比**。
- flow 臂先跑 **NFE sweep**（steps 2/4/8/16/32，8 对子集）→ 找拐点 → 全量用拐点步数。
- latent_flow 全量默认 steps=20（NFE=40）；pixel_flow 默认 steps=8（NFE=16，全量成本考虑）。

---

## 4. 预期与判读规则（写死在前面，避免事后解释）

| 假设 | 预期 | 判读 |
|------|------|------|
| H1 | `latent_reg ≥ latent_flow@任意NFE` | 成立 → 采样在这个 PSNR 任务上是净损失 |
| H2 | `pixel_flow < pixel_reg`，差 ≥ 0.5 dB | 与文献一致（pixel diffusion PSNR 弱于回归） |
| H3 | `latent_flow` 的 PSNR 随 NFE 单调上升到平台 | 平台值 ≈ H1 的上界 |
| H4 | `pixel_reg` vs E11 33.47：DiT 结构本身能否打赢 SwinIR | ±0.3 dB 内视为打平（同协议噪声级） |
| H5 | `latent_*` 不会超过 **44.74**（VAE 底噪硬顶） | 超过即为度量/对齐错误 |

**决策规则：**

- 若 `reg` 明显赢 → 结论是"**RealSR ×2 的 PSNR 主线上，扩散采样是负收益**"；
  想要"又快又像 diffusion"，下一步做 **2-Rectify / 一致性蒸馏**（用训好的 flow 模型自采样造 (noise, x₀) 对再训一轮，把 ODE 拉直），把 NFE 压到 1–2，看能否追平 reg。
- 若 `pixel_reg` 打平/超过 E11 → DiT 主干值得进回归线，做 DiT-vs-SwinIR 的同容量 A/B。
- 若 flow 反超 reg → 说明生成先验在这个数据上真的有用，值得加长训练（50k+）与放大模型（DiT-B）。

---

## 5. 开跑

```bash
# 1) 推代码（本机 PowerShell）
scp -r diffusion scripts/server ds@10.222.120.101:~/realsr/

# 2) 服务器自检（3 分钟）
ssh ds@10.222.120.101
export HF_ENDPOINT=https://hf-mirror.com
cd ~/realsr
python -m diffusion.smoke_test
find "data/RealSR(V3)" -name '*_LR2.png' | wc -l      # Train 应为 812，Test 200

# 3) 一键全矩阵（nohup，约 6–8h）
nohup bash scripts/server/run_diffusion_matrix.sh > experiments/diffusion/matrix.log 2>&1 &

# 4) 只要评估（已有 ckpt）
SKIP_TRAIN=1 bash scripts/server/run_diffusion_matrix.sh
```

脚本末尾会自动打汇总表：

```
arm            objective      n   Y-PSNR   Y-SSIM     RGB
latent_flow    flow         100   xx.xxx  0.xxxx    xx.xx
    sweep nfe= 4 n=  8 Y xx.xxx/0.xxxx
...
reference: E11 regression (ModSwinIR + Align-L1, 12k) Y 33.47 / 0.9144
```

**先跑 200 步看日志**（`ms/step` + 显存）再决定要不要调 batch；脚本每 50 步打一次。

---

## 6. 再往后（按 H1–H4 结果触发）

| 触发 | 下一步 |
|------|--------|
| reg ≫ flow | 2-Rectify / 一致性蒸馏，把 NFE 压到 1–2，追平 reg |
| pixel_reg ≥ E11 | DiT 主干进回归线；做同容量 DiT-vs-SwinIR A/B |
| latent 全线低于 33.47 | 检查：条件注入方式（concat vs cross-attn）、latent L1 vs 像素 L1（`--pixel-loss-weight`）、更长训练 |
| 有余力 | **Align-L1 迁移**：RealSR 有 1–3px 非刚性错位，回归线靠它 +0.10~0.57 dB；diffusion 目标同样可对齐后再编码 |
| 有余力 | 50k step / DiT-B / ×4 同配方 |

## 7. 不要做

- 不要用 SD1.5/SDXL VAE 跑 latent 主线（底噪 34.6–36.1，直接封顶）。
- 不要在本机 8G 上训练，只跑 `smoke_test`。
- 不要用默认 `--sample-hr 1`（已改默认，别改回去）。
- 不要把 n=30 子集的数字和全量 100 对横比（`OFFICIAL_EVAL.md` 已注明）。
