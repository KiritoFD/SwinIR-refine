# RealSR V3 ×2 — 阶段性总结（2026-09-10 → 09-19）

> 本文是全项目的**一份式**总结：任务与协议、每一条尝试及其判定（有效 / 无效 / 待定）、
> 硬件与基建结论、方法论教训、当前状态与下一步。数字全部为 RealSR V3 官方 Test.m
> **全量 100 对**（Canon 50 + Nikon 50），limited-range Y（BT.601 16–235）、uint8、
> crop=0、shave=0，除注明"12 张子集 / 16 对 val"外不可与其它口径横比。
>
> 细节分文档见文末索引；主表在 `RESULTS_SUMMARY.md`，结构说明在 `MODEL_COMPARISON.md`。

---

## 0. 一页结论

**最终成绩：Y 34.1988 / SSIM 0.9251 / MUSIQ 55.71 / MANIQA 0.347**
（`s1_b64` U-Net 18.67M + BSRGAN 预训练 + RealSR 后训练，单卡 4090 全流程 6.1h）

| 相对各锚点 | ΔY |
|---|---|
| vs 原版 SwinIR-largeish（32.97） | **+1.23** |
| vs E11 ModSwinIR+Align 旧锚点（33.47） | **+0.73** |
| vs DiT 最强 pixel_reg（33.76） | **+0.44** |
| vs bicubic 地板（31.7335） | **+2.47**（MUSIQ +16.3，MANIQA +0.069） |

**一条收益排序贯穿全程：架构 ≫ 容量 > 数据 > 步数。**

| 改动 | ΔY | 成本 |
|---|---|---|
| **stride-2 → stride-1（在目标分辨率运算）** | **+0.38** | 参数少 3.7 倍 |
| 容量 5M → 18.7M | +0.16 | 显存/算力线性涨 |
| 容量 18.7M → 72.5M | +0.05 | 参数 ×3.9 |
| 4.8h BSRGAN 预训练（3.2M 样本） | +0.09 | 墙钟 ×3 |
| 后训练步数 ×1.4 | +0.01 | 真收敛 |
| 压平通道金字塔 | **−0.07** | 负收益 |
| 门控 FFN（16 对 val 中途） | 约 −0.3 | +33% FLOPs |

**这条线已接近饱和**：容量 ×3.9 只换 +0.05，步数翻倍 +0.01，预训练 +0.09。
剩余的真瓶颈是**数据多样性**（390 对 × 约 2460 epoch，每张图被看两千多遍）。

---

## 1. 任务、协议与地板

### 1.1 任务
RealSR V3 ×2：真实相机（Canon 5D3 + Nikon D810）变焦拍摄配对数据，
Train 406 对（其中 16 对留作早停 val，实际训练 390 对），Test 100 对。

### 1.2 评估协议（所有主表数字的口径）
对齐官方 `Test.m`：RGB 训练 / **Y 测试**，MATLAB limited-range `rgb2ycbcr`，
uint8 后算 PSNR/SSIM，`modcrop 4`，**不 shave**，全量 100 对取均值。
所有模型共用同一份 `diffusion/eval_official.py`；MUSIQ / MANIQA 由 `diffusion/iqa.py`
逐图计算（`--iqa`）。

### 1.3 地板（必背）
| 地板 | Y | SSIM | RGB | MUSIQ | MANIQA |
|---|---|---|---|---|---|
| bicubic 上采样 | **31.7335** | 0.8876 | 30.01 | 39.396 | 0.2782 |
| bicubic → VAE 往返（flux1） | 31.5971 | 0.8854 | 29.89 | — | — |

第二条是 latent reg 臂零初始化时的输出 —— **任何 latent/flow 臂低于 31.60 等于连"什么都不做"都不如**，
这条线把扩散臂的判据从"比 E11 低多少"变成了"是否为负贡献"。

### 1.4 协议纪律（踩过的坑）
- **12 张子集 / 30 对子集 / 100 对全量不可横比**（子集会放大噪声：Align 的增量
  12 张口径 +0.57，官方全量 +0.10；E10 的 n=30 "33.86" 是子集偏易）。
- **full-range luma 与 limited-range Y 不可横比**（旧表 32.40 是旧口径）。
- 引用 "SwinIR on RealSR" 必须分清两种协议：(A) RealSR Train 上训练（与我们可比）、
  (B) 官方 BSRGAN 预训练零样本（数据完全不同）。官方 SwinIR real-SR **从没见过 RealSR**，
  它训在 DF2K+OST+WED（≈1.4–2 万张）+ BSRGAN 合成退化上。

---

## 2. 时间线：两条战线

| 阶段 | 时间 | 硬件 | 代码 | 内容 |
|---|---|---|---|---|
| **A · SwinIR 回归线** | 09-10 → 09-15 | 本地 RTX 4070 8GB | `model/` | 官方 SwinIR 复现 → ModSwinIR 改造 → 损失/对齐消融 → 官方协议全量评估 |
| **B · 服务器扩散/容量线** | 09-16 → 09-19 | 服务器 RTX 4090 48GB | `diffusion/` | VAE 底噪 → latent/pixel × flow/reg 2×2 矩阵 → U-Net 骨干 → stride-1 sweep → 容量/形状 sweep → 预训练 → 128 对齐加宽 → 门控 FFN |

---

## 3. 尝试清单与判定

### 3.1 ✅ 有效的（按收益排序）

**① stride-1：让整网在 HR 尺度运算 —— 最大单项，+0.38 dB，参数还少 3.7 倍**
原 U-Net 用 `in_stride=2`（stem 先降到 LR 尺度，整网在 LR 跑，最后 pixel-shuffle 抬回），
省 29% FLOPs、显存降 4.5 倍，但 bicubic 上采样后的 HR 尺度上仍有真实高频结构要重建，
pixel-shuffle 从 LR 特征"变"出高频会丢相位信息。实测：

| 配置 | 参数 | Y |
|---|---|---|
| stride-2 `1244 b64` | 18.68M | 33.56 |
| **stride-1 `s1_b32`** | **4.99M** | **33.94** |

**② 残差 bicubic + 零初始化输出头（必备）**
`ŷ = bicubic(x) + f_θ(x)`，头零初始化 → step0 精确等于 bicubic 地板（PSNR 4.6→28.8 条件化）。
同时它给了我们一个免费的诊断工具：任何臂"低于零初始化值"= 没学到东西（latent reg 就栽在这上面）。

**③ L1 替代 UWCL/CX 主损失 —— +0.95 dB（12 张口径）**
CX 类损失在 RealSR 上可迁移掉 2–3 dB，PSNR 主线必须用 L1。

**④ Offset-aligned L1（Align）—— 官方口径 +0.10 dB，旧 12 张口径 +0.57**
RealSR 残余 1–3px 非刚性错位使像素 L1 比错坐标；训练期对 HR 做搜索对齐（±3px）再罚 L1。
推理零成本。**注意**：A0（无 Align）的 MUSIQ 反而比 E11 高（50.07 vs 49.15）——
Align 换保真、牺牲一点感知，是真实的 distortion–perception 权衡。

**⑤ 容量扫描：拐点在 10M 以上 —— 5M→18.7M 共 +0.16 dB**

| base | 参数 | Y |
|---|---|---|
| 32 | 4.99M | 33.9445 |
| 40 | 7.58M | 34.0070 |
| 48 | 10.73M | 33.9946 |
| 64 | 18.67M | 34.1083 |
| 128 | 72.53M | 34.1560 |

5.0M→10.7M 完全平坦（0.06 dB），10.7M→18.7M +0.11，**18.7M→72.5M（×3.9 参数）只有 +0.048**。
容量已饱和；DiT 上同样验证过（6.28M vs 34M 只差 0.2 dB）。

**⑥ BSRGAN 预训练 + 后训练 —— +0.09 dB（有效但性价比最低）**
DIV2K+Flickr2K 3550 张 × BSRGAN 退化（σ 按 ×2 缩放，mean 30.13 dB 对齐真实退化强度）
预训练 25000 步（3.2M 样本，4.8h）→ RealSR 微调。四个指标一致小幅改善（不只是 PSNR），
收益是真的。后训练 8000→11000 只 +0.01（patience=15 自己停，真收敛）。
**为什么这么少**：390 对 × 10000 步 × batch 96 ≈ 2460 epoch，每张图被看了两千多遍 ——
缺的是退化/场景**多样性**，不是更新量；BSRGAN 合成退化与真实相机退化只部分重合。

**⑦ 更长训练 schedule（旧线）—— +0.6~0.8 dB（12 张口径）**
406 对下 4k→12k 仍明显上升；与文献 33.5–34.5 Y 的差距主要是训练预算，不是缺头。
但预算收益现已饱和（见上）。

**⑧ U-Net 替代 DiT 骨干 —— 吞吐 13 倍，这才让 sweep 变得可做**
DiT-S 的 FLOPs 有 **64% 花在全分辨率 n² attention 上**（HR128 → 4096 tokens），
对 SR 是浪费。换成 U-Net（base 64, mult (1,2,4,4), num_res 2, attn 在 level 2/3）：
18.67M 参数、940 samples/s vs DiT-S 的 70（同显存 21.7G）。两者接口一致（`forward(x,t)`），
`--backbone {dit,unet}` 可切换。

**⑨ 128 通道对齐 —— 同一张卡算力 1.8×（硬件级收益）**
b64 10.3 → b128 **18.8 TFLOPS**。参数/FLOPs 比值完全不变（0.394）——
收益来自 GEMM 不再走 padding + kernel 降级，不是存算比变好。
**b128 纯加宽（72.5M，1.9h）Y 34.1560，几乎追平 b64+4.8h 预训练的 34.1889 —— 加宽比预训练划算。**

**⑩ EMA 0.999**（最终 ckpt 用；短 schedule 下 ≈0）、**Y 通道官方评估口径**（与文献对齐）。

### 3.2 ❌ 无效 / 负收益（不要再开）

| 尝试 | 判定 | 证据 |
|---|---|---|
| **压平通道金字塔**（等比放大思路） | **−0.07 dB** | `1122_b96`：激活 +16%、存/算 +14%、参数 −38%、墙钟一样，Y 却 34.0364 < 34.1083；`1124_b80` 34.0521 同低。**U-Net 金字塔本身有用**，剩余三档中止 |
| **门控 FFN**（SpatialGatedFFN，零初始化） | **≈打平（+0.05 dB），判负关线** | 官方 100 对 34.1580 vs b64 的 34.1083，MUSIQ/MANIQA 略降；代价 +33% FLOPs、+6.25M。定位要澄清：参数/FLOPs 与堆通道严格等比（1.335 vs 1.333），**没有容量效率优势**。教训：16 对 val 的中途 −0.3 是子集噪声（官方数并不差）——**小 val 只能看趋势，定生死要用官方数** |
| **坐标注入 / 频域软路由（N1/N2）** | **≈打平/微负（−0.02 dB），不采纳** | `run_new_arms.sh`（官方 100 对）：N1 `--coord` Y 34.0848、N2 `--freq-route` Y 34.0883，vs 锚点 34.1083 均在 ±0.05 平局带内且略负，MUSIQ/MANIQA 持平微降。**打破平移不变 / 平滑区去底噪的预期卖点未兑现**，与 FFN 同命运：b64 基线已接近饱和，小机制修不出点。详 `NEW_ARMS.md` §结果 |
| **UWCL / CX 主损失** | **−0.95 dB** | 绝不用作 recon loss |
| **Wiener FFT 频带增益（E13）** | 负收益 | 31.11/31.78 vs Align 31.73/32.40；残差本身已是高频，OOM-prone |
| **AmpPhase（E14）** | 负收益 | 29.92/30.39 partial；FFT 不稳定 |
| **RadialPSF unsharp gate（E15）** | 负收益 | 30.71/31.41 @8k |
| **LP-KPN 叠在 Align 上（E12）** | 中性（≈E11） | 33.00 全量，弱于 E11 |
| **主干现代化 OCA/GDFN/AMF/FiLM/DCN** | **≈0（噪声级）** | 同容量对比 31.09 vs 31.16。该 regime 数据/目标 > 模块 |
| **EMA（短 schedule @6k）** | ≈0 | 早期还拖慢；长训/最终 ckpt 用 |
| **E5 SwinIR-classical 频域头方向** | 失败 | 31.51 / 0.8784 |
| **10M + fp16 + lr 2e-4** | NaN collapse | 用 bf16 + lr≤1e-4 + 梯度累积 |
| **batch=1 不加倍步数** | 欠训练 | 步数随有效 batch 缩放 |
| **Muon 优化器** | **未闭合** | 已移植 `--optimizer muon` 并修好稳定性（无 AMP、batch4×accum2、lr 1e-3），但 RealSR 上的正式 A/B 没跑完，**勿当已验证** |
| **latent_flow_sd / latent_flow_sdxl** | 开跑前即被底噪封死 | flux 44.7 / sdxl 36.1 / sd-ft-ema 34.6 dB（n=15），训练后实测还要掉 5 dB 以上 → 这两个臂只能落 22–24 dB，3.1h 改变不了任何结论，砍掉 |
| **Mamba / VSS 扫描骨干**（stride-1 & stride-2，多尺寸） | **样本效率远逊 U-Net，架构赌注不成立 → 关线** | 矩阵（8 臂 + U-Net 等预算对照，见 `MAMBA_MATRIX_RESULTS.md`）：**等 328k 样本** U-Net 33.97 vs Mamba s2 31.68（**−2.3 dB**）；Mamba 给到 **1.54M（4.7×）** 才 33.42，仍低于 U-Net@329k。stride-1 在 Mamba 内部确优于 stride-2（+0.89，复现「stride-1 是最大杠杆」），但绝对水位仍比 U-Net 低 ~1 dB。机制：扫描访存受限（~1–2 TFLOPS）vs U-Net 稠密 GEMM（10–19） |

### 3.3 ❌ 扩散线：在这个预算下整体打不过直接回归（重要负结果）

2×2 矩阵（latent/pixel × flow/reg），DiT-S 32.9–34.0M，flux1-vae（f8/16ch）：

| 空间 | reg（NFE 1） | flow |
|---|---|---|
| **pixel** | **33.7627** / 0.9191（XS 6.28M：33.5604） | 26.7913 / 0.5504（NFE 16） |
| **latent** | 31.1234 —— **低于自己零初始化的 31.5971** | 28.3614（NFE 40）；残差版 29.4617 |

- **所有 flow 臂全部低于 bicubic 地板**；采样代价：pixel 上 ≈4.8 dB、latent 上 ≈2.8 dB（reg−flow）。
- NFE 饱和曲线：两条 flow 在 NFE 64 收敛到**同一个值 ~29.1**（pixel 低 NFE 爬升陡，
  latent 的 ODE 更平），NFE 拉满也补不回 reg。⚠️ 两个 flow 的"官方数"NFE 不同（pixel steps=8 / latent steps=20），横比要用 NFE 对齐后的子集数。
- **latent reg 完全没学起来**：31.12 < 31.60。LR 假设已否证（3e-5 曲线与 1e-4 一模一样）；
  输出缩放扫描 α=0.25 最优也只 +0.045 dB，α=1 掉 0.47 → 学到的方向里几乎没有可用成分。
- **唯一能打的扩散臂 pixel_reg 其实是单步回归**（NFE 1），不是生成式采样。

**感知指标的转折（本项目重要的方法论收获）**：
`latent_flow_flux` PSNR 28.36（比 bicubic 低 3.4 dB）但 **MUSIQ 50.12 > E11 的 49.15**，
而 MANIQA 判它 0.2236 < bicubic 的 0.2782 —— 教科书式 **distortion–perception 分离**。
规律：**四个指标在回归类模型上排序完全一致，在生成类模型上彻底分裂**。
只用 PSNR 会把 flow 判死，只看 MUSIQ 会把模糊生成器判成最好 —— 三者必须一起报。

### 3.4 ⚠️ 判为"无效"时要小心混淆的几个教训

- **预训练验证集口径**：曾用 RealSR val 给 BSRGAN 预训练挑 `ckpt_best` ——
  那测的是域迁移不是预训练质量，等于用错误指标挑微调初始化（b40 挑中的 step 7000 纯属噪声）。
  已修：预训练 val 用预训练域留出集（DIV2K_valid，与训练零重叠）+ 确定性退化种子。
- **16 对 val 太吵**：把本该跑满的微调在 step 8000 掐断（预训练版）。
  放宽 patience 到 15 重跑后 11000 步自己停 —— 才是真收敛，多挤出 +0.01。
- **零样本迁移曲线会回落**：b40 预训练在 RealSR val 上 32.25(3k)→32.45(7k)→32.37(10k)，
  预训练越长越偏离目标域。

---

## 4. 硬件与基建结论（4090 48GB / 窗口内实测）

### 4.1 训练硬件
| 观察 | 数字 |
|---|---|
| **加大 batch 吞吐封顶** | 940 → 1067 samples/s 就不动；显存超 ~40G 后超线性变慢。**按吞吐选 batch，不按显存占用选** |
| **收窄通道省 FLOPs 反而慢 2.6×** | C=32 时 GEMM 太瘦，10.7 → 4.4 TFLOPS（−59%）。省下的 FLOPs 被执行效率下降吃掉还倒亏 |
| **128 对齐** | 10.3 → 18.8 TFLOPS（1.8×）；收益来自 GEMM 不走 padding |
| **torch.compile** | latent-S 1.26×（值得开）；pixel-S 只 1.05× 但**显存减半**（40.8→20.7G） |
| **梯度检查点** | FFN 版 32.51 → 10.19 GB（3.2×），只慢 25%；b128+FFN 不开会在 step 500 突涨 47G OOM |
| stride-1 显存 | ∝ base：b32 138 / b48 207 / b64 276 MB/样本 |

### 4.2 数据与评估基建
- **预解码缓存** `data/decoded/`（4562 张 / 31.2GB / 21s 建好）：RealSR 10.95 ms/item，解码瓶颈消除。
- **BSRGAN 退化的瓶颈是退化本身**（JPEG 往返），不是解码；更根本的是
  **OpenMP 线程爆炸**：12 worker × 24 OMP 线程 = 288 线程抢 24 核，128×128 小卷积被线程调度拖慢 14 倍。
  修法 = DataLoader `worker_init_fn` 里 `torch.set_num_threads(1)`，loader 84 → 258 samples/s。
- **像素 flow 官方评估 ~10 s/NFE/对**（100 对 @NFE16 ≈ 4h，NFE64 ≈ 10.5h 跑不起）；
  latent flow 只要 ~7 min（token 少 64 倍）。**排程前必算评估账**；
  eval 曾是全仓库唯一无 autocast 的模块 + 每图 `empty_cache()` → 5.2h 评估里 ~2.5h 是 infra 浪费。
- 大模型训练期 val 对整图做**无分块**前向会 OOM → `--eval-every 0` 或分块。
- `run_val` 用 tile 时必须按 `model.align` 对齐（stride-2 版 align=32；RealSR 1000%32≠0 会崩）。

### 4.3 远程作业流程（每次都踩的坑，已固化）
`pkill -f` 会匹配 ssh 自己的命令行 → 用 `[c]hain_afte` 自排除模式或 ps 取 PID；
杀链路要杀**进程组**；`.sh` 上服务器前必须去 CRLF（让 chain 脚本自己 `sed -i 's/\r$//'`）；
`python -u` 防 stdout 块缓冲；`kill -9` 后显存不立刻释放，启动前轮询 `nvidia-smi`；
tmux 死 session 会让链路静默跳过，启动前先 kill-session；
预训练 batch 必须按模型尺寸调（固定 256 让 b48/b64 直接 OOM）；
pyiqa 安装必须 `--no-deps`（否则拉 torch 2.14+cu13 换掉环境）；权重走
`HF_ENDPOINT=https://hf-mirror.com`，不需要 GitHub。

---

## 5. 方法论收获（比单点数字更值钱的部分）

1. **先建地板，再谈超越**：bicubic 地板 + VAE 往返地板（零初始化值）让"负贡献"立刻现形
   （latent reg 训练后反而低于地板 0.47 dB，"训练帮了倒忙"这种结论没有地板根本说不出口）。
2. **评估协议是结论的一部分**：同模型在不同子集/口径下能差出 0.5 dB 级；
   本项目全部主表统一到同一份 eval 代码 + 全量 100 对后，之前"互相矛盾"的数字全部自洽了。
3. **保真之外必须配感知指标**：MUSIQ/MANIQA 接入后才看见 A0/E11 的权衡与 flow 的分离；
   MANIQA 在极端合成图上不守规矩，只用于真实 SR 输出间的相对排序。
4. **容量不是瓶颈时，别在容量上花预算**：5M≈10.7M 平坦 + 18.7M→72.5M 只 +0.05，
   同样预算花在 stride-1（+0.38）上回报高一个数量级。
5. **归因要拆干净再下结论**：stride-2→1、batch、128 对齐、FFN 存算比等每一项都做了
   单变量对照；"FFN 提升存算比"的错误结论靠逐项复算纠回（它降低存算比，卖点是表达力）。

---

## 6. 当前状态（2026-09-19 18:00）

| 项 | 状态 |
|---|---|
| 最优模型 | `s1_b64` + BSRGAN 预训练 + 后训练 11000 步，**Y 34.1988**，ckpt 在服务器 `experiments/diffusion/s1_pretrain_b64` / `b64_ft15k` |
| 性价比最优 | `1244 b128`（72.5M，1.9h）：34.1560 |
| b64 + 门控 FFN（24.92M） | 官方 100 对 **Y 34.1580** / SSIM 0.9250 / MUSIQ 54.74 / MANIQA 0.339 —— PSNR +0.05 但感知略降，**≈打平，不值 +33% FLOPs，关线**。教训：16 对 val 的中途 −0.3 是子集噪声 + batch 混淆（FFN 96 vs 对照 128），「负收益」判早了 |
| Muon A/B | 未闭合（代码就绪） |
| **Mamba 矩阵** | **已收官（09-20，10.3h）**：等预算下 U-Net +2.3 dB，Mamba 靠 4.7× 预算仍追不上 → **架构赌注失败，关线**。详见 `MAMBA_MATRIX_RESULTS.md` |
| **三新臂 N1/N2/N3** | N1 坐标 / N2 路由 **已跑，均 −0.02 dB 平局/微负→不采纳**；N3 对抗预训练 10600/25000 中断，**本轮暂缓**（复跑须先归档半截 `adv_pretrain/`）。详 `NEW_ARMS.md` |
| **小波/等变/双射U-Net（本波）** | **已收官**。口径：主看 SSIM/MUSIQ/MANIQA。**方向二 dwt-loss 完胜且与预训练叠加**，λ 峰值=**λ8**（λ10 回落）；**交付冠军 `b64_pre_dwt8 + TTA`：SSIM 0.9279 / MUSIQ 55.83 / MANIQA 0.3516 / Y 34.38**（**vs SwinIR-largeish：SSIM +0.022 / MUSIQ +9.0 / MANIQA +0.047 / +1.40dB**）。×4 dwt 全面有效；×3 处 λ5 过锐（需 scale-adaptive λ）。方向三等变、方向一 dwt-unet 均判负。下一步：分频 2.0 → 频域 rectified flow。详 `WAVE_ARMS.md` |
| ×3 / ×4 | **已跑（zero-pretrain，plain vs dwt5）**：×4 dwt 全面有信（MUSIQ +1.10/MANIQA +0.014）；×3 dwt 只涨 PSNR、MUSIQ 反降→需 scale-adaptive λ。详 `WAVE_ARMS.md` |

**b64_ffn 官方评估（占位，待回填）**：Y — / SSIM — / MUSIQ — / MANIQA —
（对照 b64 无 FFN：34.1083 / 0.9246 / 55.218 / 0.3416）

---

## 7. 下一步建议（按优先级）

1. **数据多样性是唯一的大瓶颈**（390 对 × 2460 epoch）。方向不是更大的模型，而是：
   更强的在线增强 / 更贴近真实相机退化的合成（目前 BSRGAN 只部分重合，迁移仅 +0.09）/
   引入同域真实数据（其它相机/场景的真实配对）。
2. **FFN 判定收尾**：等官方 100 对数；若仍 >0.1 dB 落后则关线，收益排序保持"压平/门控都负"。
2b. **机制类新臂已连输两阵**（N1 坐标 / N2 路由 均 −0.02 dB 平局）：b64 基线上“针对性机制”这条也接近饱和。若还要挖，**只剩 N3（对抗数据多样性）这个未验证项**——它赌的是数据而非架构，与上述不同轴；复跑先归档半截 `adv_pretrain/` 再干净跑满 3.2M 预算。
3. **Muon A/B 闭环**：E11 或 b64 配方上跑一次，回答"优化器在这条线上有没有戏"。
4. **×3 / ×4**：stride-1 U-Net 配方直接迁移，是论文完整性最低垂的果实。
4b. **小波高频 Loss 叠加预训练**（本轮最高性价比的下一步）：λ=3 的 dwt-loss 在**零预训练** b64 上已达 34.2199，而旧冠军是**预训练+后训练** 34.1988——把 dwt-loss 加到预训练配方上微调，大概率再叠出一段新高。
5. **对标文献强基线**：按标准配方下 DF2K 完整版（当前只有 3550 张）做更长的预训练；
   或按 (B) 协议测官方 SwinIR real-SR ckpt 做零样本对照 —— 引用时先分清协议 A/B。
6. **扩散线**：除非上多步蒸馏 / 一致性模型方向，否则不再投预算；
   若做，只报「协议官方数 + NFE 饱和值」两组数，且必须带 IQA。

---

## 8. 文档索引

| 文档 | 内容 |
|---|---|
| `RESULTS_SUMMARY.md` | 5 阶段 23 行完整对照表（含 IQA） |
| `MODEL_COMPARISON.md` | 每个模型的结构说明（U-Net/DiT 逐层、零初始化位置、NFE 曲线、α 扫描） |
| `PROGRESS_REPORT.md` | 本轮完整报告（结论 + 机制解释） |
| `EVAL_PROTOCOL.md` / `OFFICIAL_EVAL.md` | 官方协议定义 / 旧线官方全量结果 |
| `REGRESSION_LINE.md` / `SWINIR_LINE.md` / `RESULTS.md` | 阶段 A：分项增量、有效改动、生产配方 |
| `NEGATIVE.md` | 阶段 A 负结果清单（UWCL/Wiener/AmpPhase/RadialPSF/AMP NaN） |
| `ABLATION_FULL.md` | 消融评估协议与矩阵脚本 |
| `DIFFUSION_PLAN.md` / `DIT_LINE.md` | 扩散 2×2 设计与判读规则 / DiT 线由来 |
| `VAE_NOISE_FLOOR_REALSR.md` | 4 个 VAE 的重建底噪（选型依据） |
| `PIXEL_DIFFUSION_MEMORY.md` | pixel diffusion 显存估算（HR256 不可行的依据） |
| `MAMBA_ARM.md` / `MAMBA_MATRIX_RESULTS.md` | Mamba/VSS 臂设计·踩坑 / **矩阵完整结果与判定（关线）** |
| `WAVE_ARMS.md` | 方向一/二/三（小波 Loss / 等变正则 / 双射小波 U-Net）工程与阶段结果 |
| `PLAN_12H.md` / `NEW_ARMS.md` / `DIT_LINE.md` | 串行计划 / 新臂 / DiT 线由来 |
| `SWINIR_DATA_PROTOCOL.md` / `BASELINE_PROTOCOL.md` / `SR_SCALING_*.md` | 数据/协议边界与 sr-scaling 移植 |
| `TRAIN_HARDENING.md` | 8GB 稳定训练加固（Muon/grad-accum/dataset） |
| `.workbuddy-ai/memory/` | 每日工作日志（全部踩坑与实测吞吐的原始记录） |
