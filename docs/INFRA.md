# Infra 与效率总表 — RealSR（4090 / 48GB 实测）

本项目全部算力利用率（MFU）与基建事实的**单一权威文档**。所有 MFU 均为 4090 真机实测，非估算。
分项实验结论见 [`../exp/FINAL_REPORT.md`](../exp/FINAL_REPORT.md)（编年体全记录），本文只管"机器/吞吐/显存/数据/调度"。

- 复现脚本：`docs/bench_mfu2.py`（U-Net 各尺寸 + DiT）、`docs/bench_mamba.py`（Mamba 两后端）、`docs/bench_mfu.py`（无 GPU 的 FLOPs 交叉复算）
- 最后更新：2026-09-27（统一 fwd+bwd 口径；Mamba 双后端分别实测）

---

## 1. 硬件与环境

| | 阶段A（本地） | 阶段B（服务器 dserver） |
|---|---|---|
| GPU | RTX 4070 8 GB | **RTX 4090 48 GB ×1**（单臂即占满 → 全部串行） |
| BF16 Tensor 峰值（dense） | — | **165.2 TFLOPS**（2:4 稀疏 ≈330；FP32 shader ≈82.6） |
| 显存带宽 | GDDR6 | **~1008 GB/s** GDDR6X |
| Roofline 拐点 I★ | — | 165.2 TF ÷ 1008 GB/s ≈ **164 FLOP/byte** |
| CPU | — | 24 核（DataLoader 线程预算受此限制） |
| 环境 | 本机 Python 3.12 + torch（仅够 smoke） | conda `harness-qwen`，torch 2.10 / cu128，nvcc 12.8（自装以编译 mamba-ssm） |
| 网络 | 通畅 | **外网受限**：`huggingface.co` 不通 → `HF_ENDPOINT=https://hf-mirror.com` |

环境相关硬约束：
- `pyiqa` 必须 `pip install --no-deps`，否则依赖解析会把 torch 换成 2.14+cu13，整环境报废。
- IQA 权重（MUSIQ/MANIQA/NIQE）经 hf-mirror 拉取；**LPIPS 权重拉不下来** → 可微感知损失只能使用已缓存的 MUSIQ/MANIQA（`--nr-loss`）。
- `mamba-ssm 2.3.2.post1` + `causal-conv1d` 由 conda nvcc 12.8 源码编译，GPU 前/反向均已验证。

---

## 2. MFU 的定义与口径（本仓库曾在此出错，务必照此引用）

**定义**：`MFU = 每次迭代的模型 FLOPs ÷ (峰值算力 × 实测 s/iter)`。计时对象是**一个完整训练步** = forward + backward + `opt.step()`，bf16 autocast。

**统一取 fwd+bwd**，理由与换算：
- conv/matmul 的反向约为前向的 2 倍 → **forward-only 口径会把训练利用率系统性低估约 3 倍**（FLOPs 比恰为 3.00×，见下）。若计时还包含 `opt.step()`，实测 achieved TFLOPS 之比会略大于 3（本例 33.1 / 10.3 = 3.2×）。
- 自校验（同一模型的两种计数）：b64 前向 47.23 → fwd+bwd 141.6 GFLOP/img = **3.00×**；b128 183.97 → 551.8 = **3.00×**；b32 前向实测 12.42 ↔ 37.2/3 = 12.4 ✓。
- 把 conv 的 forward-only 数与 Transformer 的 fwd+bwd 数并排比较，是本项目早期"DiT 61% vs U-Net 6%"这个错误结论的成因。

**计数工具**：`torch.utils.flop_counter.FlopCounterMode`，覆盖 Conv2d / ConvTranspose / Linear / matmul / SDPA 及其反向。
**不覆盖**：融合 selective-scan 核自身的 FLOPs → **Mamba 的 3.6% 是下界**（真实利用率略高）。

**吞吐换算口径**：训练吞吐（fwd+bwd）比 forward-only 探针低 3–4 倍。交付态 bs=128 实测 **b64 ≈ 0.6 s/步 ≈ 213 img/s**（≈ 30.2 TFLOPS）。早期记录的 940 / 1067 / 1083 samples/s 全部是 **forward-only 探针**，且属于 **stride-2 native-LR** 骨干（见 §6 末尾的归因更正），不可与训练吞吐或 stride-1 骨干混用。

---

## 3. MFU 总表（当前权威口径：fwd+bwd，4090 实测）

| backbone | 参数 | batch | 前向 GFLOP/img | fwd+bwd GFLOP/img | s/iter | achieved TFLOPS | **MFU** | 峰值显存 | 来源 |
|---|---|---|---|---|---|---|---|---|---|
| PixelUNet b32（stride-1） | 4.99M | 64 | 12.4 | 37.2 | 0.156 | 15.3 | **9.3%** | ~8.8GB¹ | `bench_mfu2.py` |
| PixelUNet b40 | 7.58M | 64 | 19.0 | 57.0 | 0.193 | 18.9 | **11.5%** | ~11GB¹ | `bench_mfu2.py` |
| PixelUNet b48 | 10.73M | 64 | 27.0 | 81.0 | 0.223 | 23.3 | **14.1%** | ~13GB¹ | `bench_mfu2.py` |
| **PixelUNet b64（交付骨干）** | 18.67M | 64 | 47.23 | 141.6 | 0.274 | 33.1 | **20.0%** | 17.7GB | `bench_mfu2.py` |
| **PixelUNet b128** | 72.53M | 32 | 183.97 | 551.8 | 0.308 | 57.4 | **34.7%** | — | `bench_mfu2.py` |
| **DiT-S**（HR128, patch2, 12L） | 34.0M | 16 | 534.9² | 1604.6 | 0.505 | 50.8 | **30.8%** | — | `bench_mfu2.py` |
| **Mamba s1 d96（官方 `mamba_ssm` 融合 CUDA 核）** | bench 配置³ | 16 | 63.3² | 190 | 0.518 | ≥5.9 | **≥3.6%** | 14.6GB | `bench_mamba.py` |
| **Mamba s1 d96（纯 PyTorch 扫描回退）** | bench 配置³ | 1（bs>1 即 OOM） | 104.7² | 314 | 2.51 | 0.1 | **~0.1%** | 50.3GB | `bench_mamba.py` |

¹ 由"stride-1 显存 ∝ batch"外推（见 §6 每样本显存表），非直接实测。 ² `fwd+bwd ÷ 3`，标注为推算。 ³ `bench_mamba.py` 构型为 dim96 / 2RG×3 / d_state16 / expand2 / stride-1 / bs16、**不开 ckpt**；与矩阵臂均不同配置（最接近的 `m_s1_c96` 是 d_state8 / expand1 / batch128 且**开 grad-ckpt**；`m_s2_c128_*` 是 dim128 / 4RG×4 / scale-2 / 2.75M / batch 320–512）→ 3.6% 不能直接当作矩阵臂的实测利用率（见 §12 第 2 条）。

**交付态一致性检查**：bs=128 下 0.6 s/步 → 141.6×128/0.6 ≈ 30.2 TFLOPS = **18.3%**，与 bench 的 20.0%（bs=64、AdamW）同量级；差值来自 Muon 的 Newton–Schulz 正交化与优化器路径（bench 计时内是 AdamW）。

---

## 4. 历史 MFU 数字与口径演进（旧数字全部作废，此处仅留追溯）

| 曾记录的数字 | 值 | 实际口径 | 现值（fwd+bwd） |
|---|---|---|---|
| U-Net b64 有效算力 | 10.3 TFLOPS / **6.2%** | forward-only | **33.1 TFLOPS / 20.0%** |
| U-Net b128（128 对齐后） | 18.8 TFLOPS / **11.4%** | forward-only | **57.4 TFLOPS / 34.7%** |
| U-Net b32（窄通道） | 4.4 TFLOPS（vs 10.7） | forward-only | **15.3 TFLOPS / 9.3%** |
| 未 compile 的小 DiT | 25–30 TFLOPS / ~30% | 粗估 | 30.8%（巧合一致，可保留"约 30%"的说法） |
| DiT-S 整步（含 attention） | 101 TFLOPS / **61%** | 40.6 TFLOP/step（**batch=28**）÷ 0.40s，计时与计数口径不统一 | **50.8 TFLOPS / 30.8%（batch=16）** |
| Mamba / VSS 扫描 | 1–2 TFLOPS / **0.6%–1.2%**（对外说"约 1%"） | forward-only **且走的是纯 torch 回退**（bench 误传 `backend="cuda"`，代码只认 `mamba_ssm`） | 官核 **≥3.6%**；回退 **~0.1%** |
| 历史记录“吞吐 940 / 1083 / 1067 samples/s” | — | forward-only 探针；940 = **stride-2** 1244 b64 @bs390，1083 = 1128 b64 @bs390，1067 = 1128 @bs768 | stride-1 交付骨干训练态 **213 img/s**（bs128）；同口径下 DiT-S = 31.7 img/s |

**结论**：按统一 fwd+bwd 口径，U-Net 与 DiT 的 infra 都属正常水平（20% / 31%），先前"U-Net 只跑 6%、DiT 跑到 61%"这个颠倒的对比是口径混用造成的。

---

## 5. 这些数字支撑的判断

1. **U-Net 的 MFU 完全由 GEMM 宽度决定，随通道单调上升**（b32→b128：9.3%→34.7%）。这同时**推翻了"收窄通道可省 FLOPs"**的想法：C=32 时卷积 GEMM 的 K 维只有 32，tensor core 有效算力从 10.7 掉到 4.4 TFLOPS（−59%），**反而慢 2.6 倍**——省下的 FLOPs 被执行效率的损失吃掉还倒亏。
2. **128 通道对齐 = 同卡 1.8× 算力**（b64 10.3 → b128 18.8 TFLOPS，fwd-only 口径；比值层面等效于 20.0% → 34.7%）。参数/FLOPs 比值三档不变（0.394 / 0.394 / 0.396）→ **收益来自 GEMM 不再走 padding + kernel 降级，不是存算比变好**。
3. **真正的限制是“每层 GEMM 形状”，不是模型级存算比**：U-Net 与 DiT 的模型级算术强度（12422 / 1962 FLOP/byte）都远高于 roofline 拐点 I★=164 → 按带宽判据两者都不该被访存卡住；但 U-Net 实测只有 20%，而把通道从 64 削到 32（同族形状、存算比不变）就掉到 9.3%。→ **绑住利用率的是 conv GEMM 的 K 维（=通道数）与对齐后的张量核效率**；调 batch / 拉存算比都无效，只有改架构把 FLOPs 挪到更宽更低的层才有效。
4. **高 MFU ≠ 高效率**：DiT-S 以 30.8% 的利用率，把 **11.3× 于 U-Net b64（fwd+bwd，每图 1604.6 vs 141.6 GFLOP）** 的 FLOPs 高效地花在 **占其 FLOPs 64% 的全分辨率 n² attention** 上（HR128 → 4096 tokens）。U-Net 用金字塔把注意力挤到低分辨率：同口径 fwd+bwd 实测 **U-Net b64 = 233 img/s vs DiT-S = 31.7 img/s → 7.4×**。→ 换架构挪走 FLOPs 有效，调优利用率无效。（历史数字均需改口径：“53× FLOPs”是 DiT-S 前向 ÷ **stride-2** 1244 b64 的 11.41 GFLOP/图；“~13× 吞吐”是 fwd-only 探针 940 vs 70，且 940 属于 stride-2 骨干，不是交付的 stride-1。）
5. **Mamba 线的 infra 判据**：官方融合核能训（bs16 / 14.6GB）但利用率 ≥3.6% ≪ U-Net 20%；纯 torch 回退不可训（bs>1 即 OOM、激活 50.3GB）。另一侧：stride-1（序列长 16384）实测 **~30 s/步 → 2 万步 ≈ 7 天/臂**，退到 scale-2（序列 4096）才有 0.88 s/步 @ b32 / 4GB。参数量不是原因（矩阵臂仅 **2.75M**，为 U-Net 的 1/7）→ 是**扫描的核内递推状态流量**（每位置读回上一步状态再写回，序列越长费字节越多），与 U-Net 的 GEMM 形状限制是两种不同的瓶颈。整套 8 臂矩阵花 **620 min（10.3 h）**，而正解（高频子带损失 λ8）单臂仅 **1.6 h**。
6. **墙钟账（排程用，全部实测）**：b64 零预训练 10000 步 ≈ **1.4 h**；b128 纯加宽（72.5M）≈ **1.9 h**；BSRGAN 预训练（3.2M 样本 / 4.8 h）+ 后训练全流程 ≈ **6.1 h**；小波臂 ≈ **1.6 h/臂**；Mamba 矩阵 8 臂 ≈ **10.3 h**；实验分支 campaign（15 训练臂 + DCP 推理评估）≈ **16 h**。

---

## 6. 显存、batch 与形状的解析关系

**三条公式**（3×3 卷积 U-Net，`base=64`，边长 128/64/32/16）：

```
参数量   P ≈ 9 Σ c_l²                    （与分辨率 s 无关）
计算量   F ≈ 9 Σ c_l² s_l²
激活显存 M ∝ Σ c_l s_l²                  → 算术强度 I = F / M
```

**公式已用仓库实测验证**：`M ∝ Σ c_l s_l²` 预测的激活量与记录值完全对上 —— 1244_b64 → 1901k（记录 1901k）、1122_b96 → 2212k、1124_b80 → 1884k。

**`mult=1244` 的来历**（不是"128 对齐"凑出来的，是公式推出来的）：(1,2,4,4) → 通道 64/128/256/256，**前三级 `c_l·s_l = 8192` 恒定**，于是各级 FLOPs 恰好占 **30.8% / 30.8% / 30.8% / 7.7%**（算力均衡）。旧形状 1128 = (1,1,2,8) 的 `c·s` 在 8192 与 4096 间跳动、36.89M 参数被最宽的 512 通道级独占、算力分布失衡。→ **判据：c_l ∝ 1/s_l**。

| 形状 mult | stride | base/batch | 参数 | 前向 GFLOP/img | fwd-only 探针吞吐 | 官方 Y |
|---|---|---|---|---|---|---|
| 1128 (1,1,2,8) | 2 native-LR | b64 / bs390 | 36.89M | 9.42 | 1083 smp/s（10.2 TFLOPS） | —（预研臂 `unet_1128`，未跑官方评） |
| 1244 (1,2,4,4) | 2 native-LR | b64 / bs390 | 18.68M | 11.41 | 940 smp/s（10.7 TFLOPS） | 33.56 |
| **1244 (1,2,4,4)** | **1（= 交付）** | b64 / bs128 | 18.67M | **47.23** | 218 smp/s（10.3 TFLOPS） | **34.1083** |
| 1244 (1,2,4,4) | 1 | b32 / bs256 | 4.99M | 12.42 | 366 smp/s（**4.4** TFLOPS） | 33.9445 |
| 1122 (1,1,2,2) | 1 | b96 / bs96 | 11.50M | 48.09 | 与 1244 b64 同墙钟 | 34.0364 ↓ |
| 1124 (1,1,2,4) | 1 | b80 / bs128 | 18.26M | 39.17 | — | 34.0521 ↓ |

**归因更正（全仓旧文档均待同步）**：被广泛引用的 **940 samples/s 是 stride-2 native-LR 的 1244 b64（11.41 GFLOP/图）**，不是交付的 stride-1 主干（后者的数学量是 4 倍：47.23 GFLOP/图）。因此“U-Net 比 DiT-S 快 13×（940 vs 70）”这句话里的 940 用错了骨干；同口径（fwd+bwd）的对比是 **233 vs 31.7 img/s = 7.4×**。

→ **压平通道金字塔 = 负**：1122_b96 激活 +16%、“存/算”高 14%、参数 −38%、墙钟相同，但 Y 低 0.07 → U-Net 的金字塔本身有用（拉高存算比无收益，只掉表达力）。

**“存算比”有两套口径，不可同表比较**（本项目曾混淆）：
- **① 仓库表里的“存/算”** = 激活元素数(k) ÷ 前向 GFLOP/图：1244_b64 stride-1 = 1901/47.23 = **40.2**、1122_b96 = 2212/48.09 = **46.0**、1124_b80 = 1884/39.17 = **48.1**（三者都是 stride-1 同族可直接互比；与上表 11.41 的差是 stride，不是口径）。值越大 = 越“多存少算”。
- **② 标准算术强度** FLOP ÷ byte（与 ridge point 164 同量纲）：DiT-S = 1450 GFLOP ÷ 737 MB = **1962**（= 分水岭的 11.9×）；U-Net 1244_b64 = 47.23 GFLOP ÷ 3.8 MB（bf16 激活）≈ **12422**。**两个都在分水岭之上 → 模型级“带宽不够”不成立**，因此“再提高存算比”无收益（实测也如此：1122/1124 存算比更高但掉点）。利用率差异的实际来源是 §5.3 说的 **kernel 级 GEMM 形状/张量核效率**（U-Net 20% vs DiT 31%），不是字节数。

**（近似）固定 FLOPs 下参数与激活方向相反**（Cauchy–Schwarz）：params ∝ ΣC²s（倾向集中低分辨率）、activations ∝ ΣC·s。实测 1128_b64 = 36.9M 参数 / 393k 激活 vs 1244_b32 stride-1 = 4.99M / 950k（两者前向 FLOPs 9.42 vs 12.42，差 1.3× 而非完全相等，所以只能当趋势读）。

**每样本显存（stride-1 ∝ base，实测）**：b32 **138** / b40 **172** / b48 **207** / b64 **276** MB/样本；对照 stride-2 b64 = 55.5 MB/样本。→ 交付臂 bs=128 约 35 GB（48G 卡内）。

**batch 选择按吞吐，不按显存占用**。这张 4090 在超过 ~40 GB 占用后呈超线性变慢（实测 batch 390 → 768：显存 19.1 → 37.0 GB、功耗 322 → 394 W、**吞吐 1083 → 1067 反而不涨**）：

| 臂 | 优 | 劣 |
|---|---|---|
| latent-S | b=64（23.8G，**132** smp/s） | b=112（41.0G，125 smp/s） |
| latent-XS | b=96（12.0G，**328** smp/s） | b=320（39.5G，210 smp/s） |
| pixel-S | b=28（40.8G，**29.6** smp/s） | b=24（35.1G，28.0 smp/s） |
| pixel-XS | b=32（15.7G，**62.7** smp/s） | b=64（31.2G，56.0 smp/s） |

**跨臂可比性**：锁样本预算 `batch × steps`。显存不够就降 batch，不要引入 gradient checkpointing（见 §7）。

---

## 7. 训练循环 / 优化器侧的 infra

- **混合精度**：`--amp` = bf16 autocast（`train_pixel` / `train_latent` / `train_hf_flow` 统一）。fp16 + GradScaler 路径存在但本线不用。早期 10M+fp16+lr2e-4 曾 NaN 崩，改 bf16 + lr ≤ 1e-4 + 梯度累积修好。
- **Muon（当前优化器）**：≥2D 参数走 Muon、1D 走 AdamW 的复合；**bf16 下 `GradScaler = None`**。三个必须手改的接口：① `--adv-deg` 的 φ 上升步要在 muon 分支里补写（否则对抗训练静默失效）；② **LoRA 适配头必须显式 fp32**，否则 "expected BFloat16 and float bias" 崩；③ `--muon-aux-lr` 需独立 cosine。最优：lr 5e-3 / momentum 0.95 / ns-steps 5。
- **默认值陷阱（已发生一次事故）**：`--batch` 默认 **8**、`--num-workers` 默认 **4**。Phase M 的 `run_muon_first.sh` 漏传 `--batch` → 整批臂跑在 batch=8，与 bs=128 基线不可横比，靠回查每 run 的 `args.json` 才暴露。→ **落结论前必审 `args.json`**。
- **gradient checkpointing：正式臂禁用**（computation-bound 下对已占满的卡是净亏）。两个例外：
  ① 门控 FFN 的中间张量（level-3 展开 hidden 1536、proj_in 输出 3072 通道）使 b128+FFN 稳定 32.51 GB 却在 step 500 突涨 47 GB OOM → `--grad-ckpt` 降到 **10.19 GB（3.2×），慢 25%**；
  ② **Mamba 矩阵全部臂都开了 ckpt**（`run_mamba_matrix.sh` 里 `--ssm-backend mamba_ssm --grad-ckpt` 写死），因为 4 方向扫描堆 batch 的激活太大 → 该表的样本预算对比内部自洽，但**其墙钟含重算，不能反推骨干本身的吞吐**。
- **显存兜底**：所有 campaign 脚本开头 `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`；bench 脚本逐模型 build→measure→free，OOM 自动降批。
- **`torch.compile` 实测**：latent-S b=64 0.485 → 0.384 s/step（**1.26×，值得开**）；pixel-S b=28 0.947 → ~0.90（**1.05×，几乎不加速但显存 40.8 → 20.7 GB 减半**）。规律：token 多、MLP 重的臂吃 compile；小 patch attention 的臂只省显存。`--compile-mode {default,reduce-overhead,max-autotune}`，失败自动回退 eager。
- **交付臂标准配置**（`scripts/server/run_exp16.sh`、`run_mech_best.sh` 的 `$CORE`）：
  `--backbone unet --base 64 --mult 1,2,4,4 --num-res 2 --native-lr 0 --objective reg --lr-patch 64 --batch 128 --amp --num-workers 12 --cache-data 0 --decoded-manifest data/decoded/manifest.json --warmup 500 --steps 10000 --ema 0.999 --eval-every 1000 --val-pairs 16 --patience 15 --min-steps 3000 --val-eval-steps 8 --save-every 2500 --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5`
- **骨干接口一致性**：U-Net / DiT / Mamba 三者 `forward(x, t)` 签名相同（x=(B,C,H,W)，t∈[0,1000]），`--backbone {dit,unet}` 即切；Mamba 两后端接口一致、ckpt 可互载。

---

## 8. 数据管线

- **预解码缓存 `data/decoded/`**：4562 张 / **31.23 GB** / **21 s 建好**（`scripts/server/precache_hr.py`）。`DecodedStore` 按**文件路径**索引，`store.get(path)` 零拷贝 memmap、多 worker 共享 → RealSR 现场 PNG 解码的 **10.95 ms/item** 瓶颈消除。
- **BSRGAN 退化 153 ms/item —— 瓶颈是退化本身（JPEG 往返），不是解码**。batch 390 @0.42 s 需要 929 samples/s，12 worker 只有 84 → 预训练前必须先解决。
- **OpenMP 线程爆炸**：12 worker × 24 OMP 线程 = 288 线程抢 24 核，128×128 小卷积被线程调度拖慢 **~14×**。修 = DataLoader `worker_init_fn` 里 `torch.set_num_threads(1)`（已固化在 `train_pixel.py:371`、`train_hf_flow.py:85`）→ loader **84 → 258 samples/s**。24 核机器上 pixel 臂给 8–16 worker（默认 4 偏少）。
- **静默退化坑（最坑）**：`DecodedStore` 路径不匹配时**不报错**，退回现场解 PNG —— 四个基建问题叠加曾导致 40 分钟误判为"训练卡死"。→ 启动日志必须核对缓存命中数。
- latent 缓存：`data/latents/<vae>/`（`precompute_latents.py`）；训练时用 `--cache-data 0` + decoded manifest，不把整份数据读进内存。

---

## 9. 评测侧 infra

- **唯一官方评测器** `diffusion/eval_official.py`，对齐 MATLAB `Test.m`：RGB 训练 / limited-range Y（BT.601 16–235）测试、uint8、modcrop4、不剃边、全 100 对；`ssim_y` crop=5。MUSIQ/MANIQA 由 `diffusion/iqa.py` 经 pyiqa 逐图（`--iqa`）。
- **标准评估参数**：`--tile 64 --pad 16 --tile-batch 16 --steps 8 --iqa`（TTA 版再加 `--tta`）。tile 必须按 `model.align = in_stride × 2^(levels-1) × out_scale`（=32）对齐；图片尺寸非 align 整数倍时先 pad 再裁回（RealSR 1000 % 32 = 8，不处理会 skip-concat 崩）。
- **评估耗时账（排程前必算）**：

| 臂类型 | 100 对官方评估 | 备注 |
|---|---|---|
| reg（单遍，34M DiT / b64 U-Net） | **~12 min** | 交付臂默认走这条 |
| latent flow @NFE40 | **~7 min** | tile 256，8× 下采样，token 少 64 倍 |
| **像素 flow** | **~4 h @NFE16**（9–10 s/NFE/对） | NFE64 ≈ 10.5 h，跑不起 |
| 完整评估（sweep 2/4/8/16/32 + 官方 @NFE16） | **5.2 h** | 其中约 2.5 h 曾是 infra 浪费 |

- 那个浪费的来源：`eval_official` 曾是全仓**唯一无 autocast** 的模块 + **每图 `empty_cache()`** → 后补 `_amp(bf16)` 可选与 tile 批。
- **TTA**：D4 群 8 重（rot90 × flip）前向 → 反变换 → float 均值 → 单次量化（`_tta_fwd/_tta_inv`，roundtrip 全排列精确性已验）。成本 = 8× 前向。收益：SSIM/PSNR 稳定上升、MANIQA 微升、MUSIQ 常持平或略降。
- **分块正确性门禁（先于任何烧卡）**：`scripts/server/tiling_gate.sh` 对每个官方臂比较「整图一次前向」与「分块拼接」，必须 ≥30 dB（实测 45–59 dB，差异只来自边界与 u8 量化）。它当场抓出两个真 bug：① **窗口对齐**——HAT 窗口 16，而边缘分块是 64+8=72 非 16 倍数 → reshape 直接崩，`sr_lr_tiled` 现按 `model.win` 对齐；② **变量遮蔽**——`sr_pixel_tiled` 的累加器名叫 `out`，把模型结果赋给 `out` 会让之后每个切片变空（报 “size of tensor a (0)”），一度被误判成 VDSR 架构问题。`run_official_bl.sh` 的 PHASE 0 就是这道门，不过直接 `exit 1`。
- **`--residual` 在 reg 路径必须真的生效**：三处（train forward_loss / run_val / eval sr_pixel_tiled）原先都硬编码 `pred = bicubic + model(x)`。VDSR/SRCNN 这类"直接预测图像"的网络因此被白加了一层 bicubic（−9 dB 级别的静默错误）。
- 训练期 `run_val` 若对整张 1000×1400 做**无分块**前向，大模型直接 OOM → 用 `--val-crop N`（固定中心裁剪，跨臂可比）或 `--eval-every 0`。频域 flow 的头内 AttnBlock 在全图 HF 上做 n² 注意力，曾致 38/100 图 OOM → 改纯卷积头。
- `eval_official | tee` 时 stdout 块缓冲，日志可几十分钟不刷新 → **不要用 tail 判进度**，看进程存活 + `nvidia-smi`；所有训练/评估统一 `python -u`。

---

## 10. 远程作业与调度（踩实后固化的规则）

**调度**
- 一个 campaign 一个 tmux 会话；**单卡一臂占满 → 严格串行**；`wait_gpu()` 轮询 `nvidia-smi` 显存 < 2000 MiB 才拉下一臂（间隔 15 s）。
- **臂级幂等**：`<arm>/eval_iqa/eval.json` 存在即 `SKIP` → 整条链可断点续跑。重排实验用 `tmux kill-session` 中止再改配重启。
- 后台链必须 `nohup setsid` 脱离 ssh（否则随 ssh 关闭而死）；启动器要带原子锁（`mkdir` 锁，如 `start_chain2_once.sh`）——单靠 pkill 清理极易留下 0 个或 2 个实例，而两个实例会互相 kill-session。
- 死 session 会让后续阶段**静默跳过**（`start_36h.sh` 遇同名 session 只打印提示并 `exit 0`，不是报错）→ 启动前先 `tmux kill-session`。给新 session 传变量要写进 `tmux new-session -d -s X "export A=1 && bash ..."`，别指望 `ENV=x script.sh`。

**进程**
- **不要 `pkill -f "<脚本名>"`**：ssh 远端那条 `bash -c ...` 的命令行里就含这个串 → 把自己一起杀掉（exit 255 + 空输出）。用 `[c]hain_afte` 这类自排除模式，或 ps 取 PID。杀链路要杀**进程组**。
- `kill -9` 后显存不会立即释放 → 启动前轮询 `nvidia-smi` 确认回落。
- **不要就地编辑正在运行的 `run_*.sh`**：bash 按字节偏移续读文件，改完会执行乱码。

**传输与引号**
- PowerShell → ssh → bash 多层引号/重定向极易炸 → **一律写 .sh 脚本 → scp 上服务器 → `sed -i 's/\r$//'`（Windows 侧 Write 出来是 CRLF，不去会 syntax error）→ `bash /绝对路径`**。campaign 脚本开头统一自净：`find scripts/server -name '*.sh' -exec sed -i 's/\r$//' {} +`。
- **SSH 限流**：短时间高频 ssh 会触发 kex/banner 拒绝或超时 → 冷却 ~90 s 再连（本轮多次遇到，链未受影响）。

**预训练专属坑**：`PRE_BATCH` 必须按模型尺寸调，固定 256 会让 b48/b64 直接 OOM。

---

## 11. 复现工具清单

```bash
# U-Net 各尺寸 + DiT-S 的 fwd+bwd MFU（真机；逐模型 build→measure→free，OOM 自动降批）
python docs/bench_mfu2.py --peak 165.2e12 --iters 20 --warmup 8
# Mamba 两个后端分别测（官方 mamba_ssm 融合核 vs 纯 PyTorch 回退）
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python docs/bench_mamba.py
# 无 GPU 的前向 FLOPs 独立复算（交叉核对 Conv/ConvT/Linear = 2×MAC）
python docs/bench_mfu.py --samples-per-sec 218
# 每个臂的 ms/step + 峰值显存 + forward-only ms（排 36h 计划用）
python -m diffusion.bench --iters 8
```

| 脚本 | 用途 |
|---|---|
| `docs/bench_mfu2.py` / `docs/bench_mamba.py` | MFU 权威来源（fwd+bwd，FlopCounterMode） |
| `docs/bench_mfu.py` | 前向 FLOPs 解析复算（不占 GPU） |
| `diffusion/bench.py` | 全臂 ms/step、峰值显存、forward-only 计时 |
| `scripts/server/bench_loader.py` | DataLoader 吞吐（samples/s） |
| `scripts/server/bench_unet_batch.py`、`bench_batch{,2,3}.sh` | batch–显存–吞吐–功耗扫描 |
| `scripts/server/bench_s1_sizes.py`、`bench_unet_shapes.py`、`bench_shape_ladder.py`、`bench_unet_act.py`、`bench_unet_flops.py` | stride-1 各尺寸 / 形状阶梯 / 激活 / FLOPs 探针 |
| `scripts/server/bench_gated.py` | 门控 FFN 的参数/激活/显存代价 |
| `scripts/server/precache_hr.py` | 建 `data/decoded/` 预解码缓存 |
| `scripts/server/run_*.sh` + `launch_*.sh` | 各 campaign 链（tmux + wait_gpu + 幂等 SKIP） |

---

## 12. 已知不一致与待办（诚实边界）

1. **`docs/bench_mfu2.py:37` 仍传 `backend="cuda"`**，而 `MambaCore` 只认 `"mamba_ssm"`，其他值一律落到纯 torch 路径 → 该脚本的 Mamba 行测的是回退核。表中 Mamba 数据取自 `docs/bench_mamba.py`。**修法**：改成 `backend="mamba_ssm"`，或删掉该行并修正脚本 docstring。
2. **Mamba 的 3.6% 属于 bench 构型（dim96 / 2RG×3 / d_state16 / expand2 / stride-1 / bs16，无 ckpt）**。矩阵里最接近的臂 `m_s1_c96` 是 dim96 / 2RG×3 但 d_state8 / expand1 / batch128 / **开 grad-ckpt** → 3.6% 不能直接当作矩阵臂的实测利用率；要下这个结论得按矩阵配置重测。
3. **幻灯片仍是废弃口径**：`docs/slides_new/slides/13.slide`（Mamba "MFU 0.6%–1.2%"、U-Net "10.3 TFLOPS / 6.2%"）与 `12.slide`（DiT "整步 101 TFLOPS = 峰值 61%"、"拉大 batch 940→1067"）。若要重出 PPT，需同步为 §3 的表并注明 batch 与口径。
4. **DiT 的 1962 FLOP/byte 与 101 TFLOPS 来自 batch=28 的一次换算**，与 §3 中 batch=16 的 30.8% 不同源；引用时标 batch。
5. Mamba 官核的 ≥3.6% 是**下界**（FlopCounterMode 不计融合扫描自身 FLOPs）。
6. **旧文档里“激活 ∝ 2·H²·C 决定 memory-bound”这句话已被推翻**：按模型级算术强度（§6 ②）U-Net 远高于 I★=164，不存在带宽不够；正确表述是“张量核利用率受限于 conv GEMM 形状”。同步待办：`exp/FINAL_REPORT.md` §4/§5、`docs/slides_new/`、`exp/STAGE_SUMMARY.md` §4 中引用 940 samples/s、“~13×”、“53×”、“存算比”的段落需按本文 §5/§6 修正。
7. fp32 全精度与 fp8 未测；所有 MFU 只覆盖 bf16 dense。未做 `--no-amp` 对照，量化精度影响属另一议题。
8. `hf_flow` 的 `refine_* n=62`（38 图 OOM）与 `muon_mech/n3muon/pretrain` 是过程记录，非有效对比数——不要在效率或精度结论中引用。

---

## 13. 官方基线阶梯：实测 batch / 吞吐 / 工时（`probe_batches.sh` → `batch_probe.tsv`）

排程前对**每个臂**在候选 batch（64/32/16/8/4；对照臂 256/128/64）上跑 150 步，用**相邻日志差值**算 s/iter，选「吞吐最高且显存 ≤40 GB」的那档（不是"塞得下就用"，见 §6）；steps 由统一样本预算 `batch×steps = 1.28M` 反推。

| arm（官方配置） | 参数（实测） | batch | steps | s/iter | 峰值显存 | img/s | 训练工时 |
|---|---|---|---|---|---|---|---|
| edsr_off（16RB×64F，次像素） | 1.37M | 32 | 40000 | 0.0204 | 0.8 GB | 1569 | **0.2 h** |
| vdsr_off（20 层 C=64，HR 输入） | 0.59M | 32 | 40000 | 0.0408 | 1.3 GB | 784 | 0.5 h |
| rdn_off（RDN-B D16/C8/G64） | 22.12M | 32 | 40000 | 0.2857 | 14.2 GB | 112 | 3.2 h |
| rcan_off（G10/R20/F64/r16） | 15.44M | 64 | 20000 | 0.4898 | 19.8 GB | 131 | 2.7 h |
| swinir_off（dim180，6×6，win8） | 11.75M | **8**（32/64 OOM） | 160000 | 0.1633 | 8.5 GB | 49 | 7.3 h |
| unet_ctrl（我们的臂，同配方） | 18.67M | 128 | 10000 | 0.5714 | 31.3 GB | 224 | 1.6 h |
| srformer_off（dim180，win24） | 10.40M | 16 | 80000 | 1.8776 | 32.9 GB | 8.5 | **41.7 h** |
| hat_off（HAT-L RHAG12，win16） | 40.70M | **4**（16/32/64 OOM） | 320000 | 0.4490 | 22.9 GB | 8.9 | **39.9 h** |

训练合计 ≈ **97 h**（预估，不含评估；两个大 Transformer 占 82 h）。评估每臂 = 非-TTA（~12–25 min）+ 8 重 TTA（~1.5–8 h，随骨干速度）。

**实测收锤（campaign 10-03 01:03 完成，8/8 臂，0 报错）**：预估是"跑满预算"的上界，实际被早停规则截短——
HAT-L 只用了 **6.5 h**（50k/320k 步，约 0.20M 样本）、SwinIR **6.3 h**（132k/160k 步）、SRFormer 跑了满预算 **28.2 h**（80k/80k）、RDN 3.2 h、RCAN 2.7 h、VDSR 0.5 h、EDSR 0.2 h、`unet_ctrl` 1.6 h → 整条链 09-30 12:50 → 10-03 01:03 ≈ **2.5 天**（含评估）。
**教训**：报"同预算"时必须同时报**实际消耗样本数**，否则早停会让某个臂（这里是 HAT-L）只用了 1/6 的数据却出现在表里，看的人以为它"跑满了"。表格已加"实际消耗样本"列（`docs/baseline/README.md` §1a-1）。

**三条读数**：
1. **后放大远便宜于我们基线变体的预放大形式**（同一口径 fwd+bwd）：官方 EDSR 在 LR 64² 空间跑主干 → **1569 img/s**；同一身体的预放大残差变体（HR 128² 卷积，bs64×20k 跑 1.4 h）只有 **≈254 img/s**，**≈6× 差价**（FLOPs 比也正好是 4×）。所以"官方配置"不只是精度问题，也是预算公平性的前提——同一 1.28M 样本预算下两者花的时间完全不同。
2. **窗口注意力的显存/时间由 token 数决定**：HAT-L 在 bs16 OOM、bs4 才稳（22.9 GB / 8.9 img/s）；SRFormer 窗口 24 → 576 token/窗，比 SwinIR 窗口 8 贵 ~6×（41.7 h vs 7.3 h，参数量反而更小）。
3. **对照臂 `unet_ctrl` 才是该看的比较对象**：18.67M、224 img/s、1.6 h——同等预算下我们的骨干比 HAT-L 快 25×、比 SRFormer 快 26×，这是 stride-1 金字塔（注意力放低分辨率）的直接后果。

---


## 14. 一页速查（引用时只用这几个数）

| 项 | 值 |
|---|---|
| 峰值 / 拐点 | bf16 dense **165.2 TFLOPS**；I★ ≈ **164 FLOP/byte** |
| MFU 口径 | **fwd+bwd**（fwd 只占 1/3）；FlopCounterMode；Mamba 官核为下界 |
| U-Net stride-1 | b32 **9.3%** → b40 11.5% → b48 14.1% → b64 **20.0%** → b128 **34.7%**（单调，由 GEMM 宽度决定） |
| DiT-S | **30.8%**（1604.6 GFLOP/img fwd+bwd，其中 64% 是全分辨率 n² attention） |
| Mamba | 官核 **≥3.6%**；纯 torch 回退 **~0.1%** 且 bs>1 即 OOM → 不可训 |
| 交付骨干 | stride-1 b64，18.67M，前向 47.23 / fwd+bwd **141.6 GFLOP·img⁻¹**，bs128 → **0.6 s/步 ≈ 213 img/s ≈ 18.3%** |
| 带宽 | 显存带宽不是限制（两骨干模型级算术强度均 ≫ I★）；限制在 conv GEMM 的 K 维与对齐 |
| 对齐 | 128 通道对齐 → 同卡 **1.8×** 算力；窄通道 C=32 → 慢 **2.6×** |
| batch | 按吞吐选、不按显存占用选；>~40 GB 后超线性变慢；跨臂锁 `batch×steps` |
| 每样本显存 | stride-1：b32 138 / b48 207 / **b64 276** MB；stride-2 b64：55.5 MB |
| compile | latent **1.26×**；pixel 1.05× 但显存 **减半** |
| 数据管线 | `data/decoded/` 4562 张/31.2 GB/建 21 s；`set_num_threads(1)` 使 loader **84→258** samples/s |
| 评估 | reg 100 对 **~12 min**；像素 flow **~4 h@NFE16**；latent flow ~7 min；TTA = 8× |
| 禁行 | 正式臂禁 grad-ckpt（例外：门控 FFN、Mamba 臂）；单卡串行；幂等靠 `eval.json` |

