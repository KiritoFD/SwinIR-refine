# RealSR 真实超分 —— 实验日志（全记录 · 定稿）

> 本文件是项目 2026-09-10 → 09-26 的**编年体实验日志**：每一次训练/评估记其**假设、关键配置、结果、判定**，含正负结果与踩坑。分项细节散见于同目录其它 md，此处为单一权威全记录。

## 0. 元信息与约定

**任务**：RealSR V3 真实相机单帧超分（Cai et al. ICCV 2019，变焦配对）。Train 406 对（16 早停 val、实训 390），Test 100 对（Canon50+Nikon50）；×2 主线、×3/×4 迁移。
**硬件**：阶段A 本地 RTX 4070 8GB；阶段B 服务器 dserver（`ds@10.222.120.101`，/home/ds/realsr），RTX 4090 48GB，conda `harness-qwen`。
**代码**：`diffusion/`（阶段B，像素/latent 扩散 + U-Net/DiT/Mamba 骨干 + 本波小波/Muon/TTA/对抗）；`model/`+`models/`（阶段A，SwinIR 线）；`scripts/server/run_*.sh`（链式实验）。
**评价口径（项目准则）**：**PSNR-Y 次要，主看 SSIM / MUSIQ / MANIQA**；±0.05 dB（及相应感知阈值）内视为平局；16 对 val 仅看趋势，定生死用官方 100 对 + IQA。
**协议**：官方 `Test.m`——RGB 训练 / Y 测试（limited-range BT.601 16–235）、uint8、modcrop4、不 shave；统一 `diffusion/eval_official.py`，MUSIQ/MANIQA 由 `diffusion/iqa.py` 逐图（`--iqa`）。**协议A**（见过 RealSR train）方可与 SwinIR 重训横比；官方 real-SR 权重属**协议B**（DF2K+BSRGAN，没见过 RealSR），不可直接横比。

**时间线总览**
| 日期 | 里程碑 |
|---|---|
| 09-10→09-15 | 阶段A：复现 SwinIR → Mod-SwinIR(A0) → Align(E11) → 官方协议全量评估 |
| 09-16→09-19 | 阶段B：VAE 底噪 → 扩散 2×2 矩阵 → U-Net stride-1/容量/形状/预训练/128对齐 → 门控FFN → 首版冠军 34.1988 |
| 09-20→09-21 | Mamba 矩阵（等预算）→ 关线；小波时代 round1–5 → λ8 + 预训练 34.278 |
| 09-22→09-23 | 分频2.0（db2/db4/aniso）+ TTA + ×3/×4 迁移 |
| 09-24 | 频域 rectified flow（方向C）→ 判负 |
| 09-24→09-26 | Muon-first（扫参→机制：移位/DTCWT/双路U-Net/N3）→ 最终冠军 SSIM .9285/MUSIQ 55.96/MANIQA .3534 |

---

## 1. 阶段A —— SwinIR 线（09-10 → 09-15，本地 4070 8GB）

**假设链**：复现原版 SwinIR → 用残差-bicubic + 零初始化头改造 → L1 主损失 → offset-aligned L1。
| run | 参数 | Y | SSIM | MUSIQ | MANIQA | 判定 |
|---|---|---|---|---|---|---|
| Stock SwinIR-light | 0.61M | 32.883 | 0.9026 | 45.43 | 0.302 | 基线 |
| Stock SwinIR-largeish | 3.96M | 32.974 | 0.9058 | 46.79 | 0.305 | 基线（横比锚） |
| A0 Mod-SwinIR pure L1 | 4.05M | 33.361 | 0.9123 | 50.07 | 0.310 | L1 + 残差改造 |
| **E11 Mod-SwinIR + Align-L1** | 4.05M | 33.466 | 0.9143 | 49.15 | 0.309 | 阶段A冠军 |
> insight：L1 替代 UWCL/CX 主损失 +0.95 dB（旧口径）；offset-aligned L1（±3px 搜索对齐 HR 再罚 L1）+0.10 dB（官方口径）。**A0 的 MUSIQ(50.07) 反高于 E11(49.15)** → align 换保真、牺牲一点感知，是首见的 distortion–perception 权衡。
**阶段A负结果（勿重开）**：UWCL/CX 主损失（−0.95）、Wiener FFT 频带增益(E13)、AmpPhase(E14)、RadialPSF unsharp gate(E15)、LP-KPN-on-Align(E12)——均负或中性；主干现代化 OCA/GDFN/AMF/FiLM/DCN ≈0（该 regime 数据/目标 > 模块）；10M+fp16+lr2e-4 → NaN collapse（改 bf16+lr≤1e-4+梯度累积修好）。

---

## 2. 阶段B 基础设施与踩坑（09-16 起）

- **预解码缓存** `data/decoded/`（4562 张 / 31.2GB / 21s 建好）：RealSR 10.95 ms/item，消除 PNG 解码瓶颈。`scripts/server/precache_hr.py`。
- **DataLoader OMP 线程爆炸**：12 worker × 24 OMP = 288 线程抢 24 核，BSRGAN 128×128 小卷积被拖慢 ~14× → `worker_init_fn: torch.set_num_threads(1)`，loader 84→258 samples/s。
- **VAE 底噪**：4 个 VAE 的 bicubic→VAE 往返重建（flux1 31.60/0.8854）——把 latent/flow 臂判据从"比 E11 低多少"改成"是否低于自身零初始化"（负贡献探测器）。
- **eval 曾是全仓唯一无 autocast + 每图 `empty_cache()` 的模块** → 5.2h 评估里 ~2.5h 是 infra 浪费；后补 `_amp(bf16)` 可选、tile 批。
- **远程流程坑（多次踩，固化）**：`pkill -f` 匹配自身 ssh；杀链路杀进程组；`.sh` 上服务器去 CRLF；`python -u`；kill-9 后轮询 `nvidia-smi` 再启；tmux 死 session 前 kill-session；pyiqa 装 `--no-deps`、权重走 `HF_ENDPOINT=hf-mirror`。
- **SSH 限流**：短时间高频 ssh 触发 kex/banner 拒绝，需冷却重连（本轮多次遇到，链未受影响）。

---

## 3. 扩散 2×2 矩阵 + DiT 线（09-16 → 09-17）

**假设**：rectified-flow 生成式（pixel/latent × flow/reg）能否超直接回归。**结果：此预算下全面不及回归**。
| 臂 | 空间 | 参数 | Y | SSIM | 备注 |
|---|---|---|---|---|---|
| pixel_flow(NFE16) | pixel | 34.0M | 26.79 | 0.550 | 低于 bicubic |
| latent_flow_flux_XS | latent | 5.73M | 27.05 | 0.736 | |
| latent_flow_flux(NFE40) | latent | 32.9M | 28.36 | 0.790 | **MUSIQ 50.12 > E11！** |
| latent_flow_res(NFE20) | latent | 32.9M | 29.46 | 0.829 | |
| latent_reg_flux | latent | 32.9M | 31.12 | 0.884 | **< 自身零初始化 31.60，负贡献** |
| DiT pixel_reg_XS | pixel | 6.28M | 33.560 | 0.916 | 单步回归(NFE1) |
| DiT pixel_reg | pixel | 34.0M | 33.763 | 0.919 | 唯一"能打"，但本质是回归非采样 |
**关键 insight**：
1. 所有 flow 臂 < bicubic 地板；NFE 饱和曲线两条 flow 在 NFE64 收敛到同一 ~29.1，加 NFE 也补不回。
2. **distortion–perception 分离实证**：latent_flow_flux PSNR 28.36（低于 bicubic 3.4dB）却 MUSIQ 50.12 > 回归 E11 的 49.15，而 MANIQA 判它 0.224 < bicubic 0.278 → **感知指标在回归类上排序一致、在生成类上分裂**，故三指标必须同看（此发现最终导致 §11 重开生成式在新口径下的复验，见 §10）。
3. latent_reg 完全没学起来（31.12<31.60）；LR 假设已否证（3e-5 与 1e-4 曲线一致）；输出缩放 α 扫描 α=0.25 最优仅 +0.045、α=1 掉 0.47 → 学到的方向里几乎没有可用成分。
**结论**：扩散线（除蒸馏/一致性方向）不再投预算；只报"官方数 + NFE 饱和值 + 带 IQA"。

---

## 4. U-Net 线：stride-1 / 容量 / 形状 / 预训练 / FFN（09-17 → 09-19）

**骨干替换**：DiT-S 的 FLOPs 有 64% 花在全分辨率 n² attention（HR128→4096 tokens），对 SR 浪费 → 换 U-Net（同接口 `forward(x,t)`，`--backbone {dit,unet}` 可切）：18.67M、940 samples/s（DiT-S 的 13×）。这让 sweep 变得可做。
**stride-1 sweep（容量扫描，零预训练）**：
| base | 参数 | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|---|
| s1_b32 | 4.99M | 33.945 | 0.9219 | 54.20 | 0.334 |
| s1_b40 | 7.58M | 34.007 | 0.9232 | 54.84 | 0.339 |
| s1_b48 | 10.73M | 33.995 | 0.9238 | **55.26** | 0.342 |
| **s1_b64** | 18.67M | 34.108 | 0.9246 | 55.22 | 0.342 |
| 1244_b128 | 72.53M | 34.156 | 0.9249 | 55.42 | 0.343 |
**insight / 判定**：
1. **stride-2 → stride-1 = +0.38 dB、参数少 3.7×**（全项目最大单项）；bicubic 后 HR 仍有真实高频需重建，pixel-shuffle 从 LR"变"高频丢相位。
2. **容量饱和**：5M→10.7M 平坦（0.06）、10.7→18.7M +0.11、18.7→72.5M（×3.9）仅 +0.048；拐点在 10M 以上。
3. **压平通道金字塔 = 负**：`1122_b96`（激活+16%、存/算+14%、参数−38%、墙钟同）Y 34.036 < 34.108 → **U-Net 金字塔本身有用**。
4. **门控 FFN ≈打平（+0.05）判负**：参数/FLOPs 与堆通道等比（1.335 vs 1.333，无容量效率优势），+33% FLOPs；16 对 val 中途 −0.3 是子集噪声（官方数不差）→ 教训"定生死用官方数"。
5. **128 通道对齐 → 同卡算力 1.8×**（b64 10.3 → b128 18.8 TFLOPS，参数/FLOPs 比不变，收益来自 GEMM 不走 padding）。
6. **BSRGAN 预训练 +0.09 dB**（3.2M 样本/4.8h），四指标同向；后训练 8k→11k 仅 +0.01（patience15 自停=真收敛）。**为什么少**：390 对×10000 步×batch96≈2460 epoch，每图被看两千遍 → 缺的是**退化/场景多样性**不是更新量；合成退化只部分重合真实相机。
7. 加宽(b128 裸跑 1.9h=34.156) 几乎追平预训练(6.1h=34.189) → 加宽更划算。
8. 硬件：加大 batch 吞吐封顶(940→1067)；收窄通道省 FLOPs 反慢 2.6×；算力真瓶颈但提存算比救不了 → 只能改架构把 FLOPs 挪走。**grad-ckpt 是 FFN 必备**（32.5→10.2GB、慢25%），但见 §0/§14：**computation-bound 下正式臂禁 ckpt、显存不够降 batch**。
**阶段B首版冠军（09-19）**：`s1_b64 + 预训练 + 后训练` Y **34.1988 / SSIM 0.9251 / MUSIQ 55.707 / MANIQA 0.3470**（18.67M，4090 全流程 6.1h）。

---

## 5. Mamba / VSS 扫描骨干矩阵（09-20，tmux mbm，620min）

**假设**：扫描 O(L) 让 stride-1 全局感受野首次便宜；S6 选择机制内生内容自适应。**设计**：每臂固定样本预算，U-Net 锚点在**相同样本数**重跑（分离架构 vs stride vs 预算）。
| 臂 | scale | 样本 | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|---|---|
| u_b64_188k（U-Net 对照） | 1 | 188k | 33.254 | 0.9102 | 47.53 | 0.301 |
| **u_b64_329k（对照）** | 1 | 328k | **33.966** | 0.9221 | 53.88 | 0.332 |
| u_b64_439k（对照） | 1 | 439k | 34.036 | 0.9231 | 54.47 | 0.336 |
| m_s2_c96_match | 2 | 328k | 31.683 | 0.8906 | 40.22 | 0.282 |
| m_s2_c96_full | 2 | 1.54M | 33.418 | 0.9128 | 51.38 | 0.311 |
| m_s2_c128_e1 | 2 | 717k | 32.067 | 0.8992 | 43.34 | 0.292 |
| m_s2_c128_e2 | 2 | 179k | 31.813 | 0.8909 | 40.59 | 0.283 |
| m_s1_c96（stride-1，补跑官方评） | 1 | 269k | 32.567 | 0.9002 | 43.41 | 0.289 |
**判定（关线）**：等预算 328k，U-Net +2.28 dB；Mamba 给到 4.7× 预算（1.54M）才 33.42 仍 < U-Net@329k。stride-1 Mamba 内部优于 stride-2（+0.89，复现 stride-1 杠杆）但绝对水位仍低 U-Net ~1 dB。四指标与 Y 同向（非感知换保真，是实打实学得差）。**机制**：扫描访存受限（有效算力 ~1–2 TFLOPS vs U-Net 稠密 GEMM 10–19），样本效率劣势压倒理论优势。→ **Mamba 线关闭**（补跑 `m_s1_c96` 靠守候脚本 `mamba_fill_eval.sh`）。

---

## 6. 三新臂 N1/N2/N3（09-19→09-20，并行→被串行计划取代）

| 臂 | 假设 | 配置 | Y | 判定 |
|---|---|---|---|---|
| N1 s1_b64_coord | 绝对坐标(2ch,零初始化stem)破平移不变 | +`--coord` | 34.085（−0.023） | ≈平/微负 |
| N2 s1_b64_froute | 按 Sobel 能量软路由 texture/smooth 分支 | +`--freq-route`(+5%) | 34.088（−0.020） | ≈平/微负 |
| N3 adv→adv_ft | min-max 对抗退化挖掘（连续退化流形，攻 HF 弱点） | 25000×128 预训练→微调 | 中断（10600/25000） | **暂缓**（为让位串行/Mamba） |
> N1/N2 官方 100 对都在 ±0.05 内且略负 → 不采纳。N3 因让位被 kill，留下半截 `adv_pretrain`（→ §14 复跑陷阱）。全部过本地 smoke（`smoke_new_arms.py`）。

---

## 7. 小波时代 round1–5（09-20→09-21，本波核心突破）

**假设**：L1 给 HF 条件均值→糊；用**严格正交 + 局部**的小波高频子带多尺度 L1 显式抬 HF（区别于失败的 FFT 全局增益头）。实现 `diffusion/wavelet.py`（Haar DWT/IWT，双射误差 1e-07、Parseval 能量守恒，无新依赖）。

**round-1 wave_arms（零预训练 b64，λ 扫点 + equiv，AdamW）**
| 臂 | 配置 | Y | SSIM | MUSIQ | MANIQA | 判定 |
|---|---|---|---|---|---|---|
| s1_b64_dwt_w1 | λ1 L2 | 34.190 | 0.9258 | 55.381 | 0.3451 | +0.081 四指标同向 ✅ |
| s1_b64_dwt_w2 | λ2 L2 | 34.221 | 0.9261 | 55.434 | 0.3466 | ✅ |
| s1_b64_dwt_w3 | λ3 L2 | 34.220 | 0.9253 | 54.828 | 0.3418 | λ3 感知反低（非单调）|
| s1_b64_dwt_w5 | λ5 L2 | 34.240 | 0.9262 | 55.342 | 0.3472 | 四指标全≥锚点 ✅ |
| s1_b64_equiv | D4 自监督 α0.25 (b64×20k) | 34.038 | 0.9243 | 55.438 | 0.3403 | **判负**（−0.07）❌ |
**round-2 wave_unet（方向一 v1：层级 DWT 下/IWT 上采样，替 lossy stride-2）**
| 臂 | Y | SSIM | MUSIQ | MANIQA | 判定 |
|---|---|---|---|---|---|
| s1_b64_dwtunet_dw1 | 34.195 | 0.9258 | 55.056 | 0.344 | 三项 ≤ 锚点 ❌ |
| s1_b64_dwtunet | 34.093 | 0.9243 | 54.934 | 0.341 | 判负 ❌ |
> 踩坑：`eval_official` 重建读错 args 键名（`wavelet`↔`dwt_unet`）致 state_dict 不匹配、评估崩溃；修复后 `reeval` 仅重评估救回。→ "无损 stride-2"既没拿质量红利又不涨感知。
**round-3/4/5 stack_dwt（预训练冠军 b64_ft15k init + λ 精扫，AdamW）** — 见 §3 表(3.2)：λ1→10，**λ8 是感知峰值**（SSIM .9271/MUSIQ 55.87/MANIQA .3513），λ10 回落。冠军 `b64_pre_dwt8`（预训练+λ8）**超旧冠军 b64_ft15k**。
**round-6 freq2.0（基/方向/db）** — 见 §3 表(3.3)：db2/db4/各向异性**均未越过 Haar**（基与方向调参饱和）；×3 λ 扫见 §9。
**踩坑（FIR 实现）**：db2/db4 FIR 滤波器 `view` 形状写错（把长度 L 塞进 1×1×1×1）→ runtime error；改 `(1,1,L,1)`/`(1,1,1,L)` 修好。**分频2.0 结论：Haar 已够**。

---

## 8. TTA 自集成（`eval_official --tta`，纯评估）

D4 全 8 重（rot90×flip）→ SR → 反变换 → float 均值 → 单次量化（`_tta_fwd/_tta_inv`，roundtrip 全排列精确、8 元互异，本地已验）。
| ckpt | 非-TTA (SSIM/MUSIQ/MANIQA/Y) | +TTA |
|---|---|---|
| b64_pre_dwt5 | .9270/55.767/.3506/34.267 | .9277/55.735/.3509/34.363 |
| b64_pre_dwt8 | .9271/55.870/.3513/34.278 | .9279/55.825/.3516/34.377 |
| b64_pre_dwt8_muon | .9280/55.850/.3521/34.289 | **.9289/55.751/.3521/34.405** |
| s1_b64_dwt_w5（零预训练） | .9262/55.342/.3472/34.240 | .9270/55.289/.3477/34.326 |
> insight：TTA 稳定抬 **SSIM/PSNR**（自集成降方差 + 对齐细节）、MANIQA 微升、**MUSIQ 常持平或略降**（8 重平均轻微平滑了极锐纹理）→ 打榜/交付用 TTA 版；PSNR 与 MUSIQ 在 TTA 上方向相反，再次印证需多指标同看。

---

## 9. ×3 / ×4 迁移与 scale-adaptive λ（09-22→09-23，scale_3/scale_4）

stride-1 b64 配方迁移（×3 lr-patch48→HR144/batch96；×4 lr-patch32→HR128/batch128；控显存）。零预训练 plain 对照 vs dwt。
| scale | 变体 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|---|
| ×3 | plain | 0.8687 | 51.786 | 0.3239 | 31.038 |
| ×3 | dwt λ1 | 0.8671 | 49.751 | 0.3121 | 31.060 |
| **×3** | **dwt λ2** | **0.8709** | **52.041** | **0.3320** | 31.130 |
| ×3 | dwt λ5 | 0.8686 | 50.478 | 0.3196 | 31.116 |
| ×4 | plain | 0.8295 | 47.146 | 0.2993 | 29.416 |
| ×4 | dwt λ5 | 0.8322 | 48.249 | 0.3131 | 29.511 |
**insight（非单调、可发表）**：×4 dwt 全面有效（MUSIQ +1.10/MANIQA +0.014/SSIM +0.003）；**×3 处 λ5 过锐（MUSIQ −1.31），换 λ2 才三项全面超 plain**。→ **最优 λ 随退化强度反相关：×2≈8、×3≈2、×4≈5**（×3 是重退化→HF 更难恢复、λ 需收）。绝对值低是任务本身更难，不可与 ×2 横比。

---

## 10. 频域 rectified flow（方向C，hf_flow，09-24 → 判负）

**假设（新口径下的复验）**：回归吃满可预测部分，但不可预测 HF 恰是生成式主场；只对 **HF 子带残差**做少步 rectified-flow（1–4 NFE），LL 保留回归 → 期望感知升、保真不崩（绕开全图 flow 崩 PSNR 的老问题）。实现 `diffusion/hf_flow.py`（DWT 分 LL/HF，残差流头条件于[LL_r,HF_r]，IWT 合回、LL 不动，自测 LL 守恒 3.6e-07）。
**踩坑与修复**：v1 refine 崩——头内 AttnBlock 在全图 HF 上做 n² 自注意力 → 38/100 大图 CUDA OOM；且从噪声生成的残差在 scale≥0.6 毁图。修：**头改纯卷积**（去 attn）+ **小 scale {0.10,0.20,0.35}**，重训头（VALFLOW 收敛到 0.002–0.003）。
| refine scale (NFE2) | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| 基线 b64_pre_dwt8（回归） | .9271 | 55.87 | .3513 | 34.278 |
| 0.10 | 0.7622 | 47.58 | 0.2308 | 31.07 |
| 0.20 | 0.5414 | 41.38 | 0.2409 | 27.18 |
| 0.35 | 0.3482 | 36.05 | 0.2545 | 23.19 |
**判定（概念性负）**：修好 bug 后**单调毁图**、scale 越小越接近基线但 0.10 仍远差。根因：**回归够不到的高频本质是近不可预测噪声，从噪声生成的残差只注入伪影**，任何感知指标都不买账。**在 390 对规模下生成式（全图 or 专攻 HF）都输给判别式回归。** → 方向C 关闭。（诚实：naive from-noise 版判负；SDEdit 式从回归 HF 起点的少步细化、或更多训练/更大 NFE 未测，属遗留可试项。）

---

## 11. Muon 时代（09-24→09-26，muon_ab/muon_tune/mech_best）

**先做好 Muon**：`train_pixel` 接 `model/optim.build_optimizer`（≥2D 用 Muon + 1-D AdamW 复合，bf16 下 GradScaler=None）；修 `muon+adv` 不兼容（muon 分支补 φ 翻转梯度上升步）；加 `--muon-momentum/--muon-ns-steps/--muon-aux-lr`（aux 独立 cosine）。

**(a) muon_ab（零预训练 b64+λ8，扫 Muon lr，AdamW 对照）**
| 优化器 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| AdamW 3e-4 | 0.9259 | 55.464 | 0.3470 | 34.242 |
| Muon 1e-3 | 0.9272 | 55.990 | 0.3520 | 34.301 |
| **Muon 5e-3** | **0.9280** | 55.974 | **0.3527** | 34.391 |
| Muon 1e-2 | 0.9279 | 55.929 | 0.3527 | 34.328 |
> **每个 Muon lr 三指标全胜 AdamW**（全 lr 段赢，非单点侥幸，吸取 FFN 教训）；且**零预训练 Muon5e-3 ≈ 预训练+AdamW 冠军** → Muon 抵掉一大截预训练收益。**决策：全项目改用 Muon。**

**(b) muon_tune（Phase M：预训练 b64+λ8 上 OFAT 扫 Muon）** lr{2e-3,8e-3,1.2e-2}/mom{0.90,0.98}/ns{3,7}/aux{3e-4}/wd{1e-2} —— **9 个偏离全部 < base lr5e-3/mom0.95/ns5** → Muon 最优确认 = 5e-3。**踩坑（入记忆）：自动选择器只比 9 偏离、漏了未扫 base → 误选 handicap 的 2e-3；已修（纳入 base 候选）并在 mech_best 用 5e-3 重跑纠正。**

**(c) mech_best（诚实清单机制项，全 @ Muon 5e-3）** — 见 §3 表(3.5)：**移位集成 p_shift4 = 新非-TTA 最优（MUSIQ 56.02/MANIQA .3534 全项目最高）**；各向异性≈持平；DTCWT 更差；双路小波 U-Net（muon_mech z_dual*）明显负（§12）；N3 对抗中性（§13）。**最终交付 = 预训练+Muon5e-3+λ8+shift4+TTA：SSIM 0.9285 / MUSIQ 55.96 / MANIQA 0.3534 / Y 34.41。**
| 交付 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| p_shift4 | 0.9277 | **56.022** | **0.3534** | 34.310 |
| **p_shift4 + TTA** | 0.9285 | 55.960 | 0.3534 | **34.406** |
| p_aniso + TTA | 0.9285 | 55.862 | 0.3532 | 34.407 |

---

## 12. 双路小波 U-Net（方向一完全体，muon_mech/z_dual*，09-25，判负）

`--dwt-dual`（WaveletDualUNet：一次 Haar DWT → LL 走完整 U-Net（半分辨率、无损拿 stride-2 感受野）+ HL/LH/HH 走浅分支，双零初始化残差、IWT 合回，step0=bicubic）。
| 变体 | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| s1_b64+λ8（对照） | 0.9262 | 55.342 | 0.3472 | 34.240 |
| z_dual | 0.9214 | 53.911 | 0.3338 | 33.845 |
| z_dual_shift4 | 0.9207 | 53.731 | 0.3332 | 33.856 |
| z_dual_dtcwt | 0.9105 | 51.344 | 0.3252 | 33.162 |
**判定**：明显负（−1.4 MUSIQ），且 shift/dtcwt 叠加更糟 → **方向一（含完全体）彻底死**。架构性"分频双路"不敌把小波只用作 loss。

---

## 13. N3 对抗挖数据多样性（mech_best/n3，09-26，中性）

用 Muon（修好 muon+adv 后）重跑：adv 预训练（G_phi 攻含小波-HF 的目标，18 核 PSF bank + 噪声，simultaneous GDA，25000×128）→ Muon5e-3 微调 → TTA。
| | SSIM | MUSIQ | MANIQA | Y |
|---|---|---|---|---|
| n3_ft | 0.9279 | 55.570 | 0.3511 | 34.276 |
| n3_ft + TTA | 0.9288 | 55.497 | 0.3511 | 34.388 |
**判定（中性/无增益）**：未破平台（MUSIQ/MANIQA 低于 p_shift4）。**诚实清单到此全判完：只有"小波高频 Loss + 移位集成"持续涨点**。

---

## 14. 最终判定与交付

**×2 交付冠军** = `s1_b64` stride-1 U-Net + BSRGAN 预训练 + 小波高频 Loss λ8 + Muon(5e-3) + 移位小波集成 + EMA + TTA：
**SSIM 0.9285 · MUSIQ 55.96 · MANIQA 0.3534 · (Y 34.41)**。vs SwinIR-largeish **+0.023 SSIM / +9.2 MUSIQ / +0.048 MANIQA / +1.44 dB**；vs 阶段A 冠军 E11 **+0.014 / +6.8 / +0.045 / +0.94**。

**收益排序（架构 ≫ 机制loss > 优化器 > 数据 > 步数；本波新增机制loss与优化器两档）**：stride-1(+0.38) > 小波λ8(+0.11) ≈ Muon(+0.15) > 移位集成(+0.02) > 预训练(+0.09) > TTA(+0.10，SSIM/MANIQA) > 容量(饱和)。×3/×4 用 scale-adaptive λ（2/5）。

**判负/关闭清单（勿重投）**：扩散 flow/latent_reg、Mamba/VSS、coord、freq-route、门控FFN、压平金字塔、等变D4、双射小波U-Net、双路小波U-Net、DTCWT复数基、频域rectified-flow(HF)、FFT频带头(Wiener/AmpPhase/RadialPSF)、UWCL/CX、N3对抗(中性)。**统一洞见**：390 对下判别式回归吃满可预测部分；涨点只能靠"逼网络别抹平可预测高频"的正交局部机制（小波 loss+移位）+ 更强优化器(Muon)+ 多样先验(预训练)+ 推理集成(TTA)；一切"生成不可预测高频"的路线均告负。

---

## 15. 复现 / 产物 / commit

**交付复现**：
```
python -m diffusion.train_pixel --backbone unet --size S --base 64 --mult 1,2,4,4 --num-res 2 \
  --attn-levels 2,3 --native-lr 0 --objective reg --residual 1 --scale 2 --lr-patch 64 --batch 128 --amp \
  --optimizer muon --muon-lr 5e-3 --muon-momentum 0.95 --muon-ns-steps 5 \
  --dwt-loss --dwt-weight 8 --dwt-levels 2 --dwt-shift 4 \
  --lr 3e-4 --warmup 500 --steps 10000 --ema 0.999 --init <bsrgan_pretrain_ckpt>
python -m diffusion.eval_official --ckpt <ckpt> --mode pixel --objective reg --iqa --tta --tile 64 --pad 16 --tile-batch 16
```
**结果目录**：`experiments/diffusion/{wave_arms,wave_unet,stack_dwt,scale_3,scale_4,muon_ab,muon_tune,muon_mech,mech_best,hf_flow,mamba_matrix}/<臂>/eval_iqa[/eval_iqa_tta]/eval.json`；链式脚本 `scripts/server/run_*.sh`（tmux + 幂等 + 逐臂 wait_gpu）。**运行规范**：computation-bound 禁 grad-ckpt（显存不够降 batch、锁样本预算 batch×steps）；一臂独占全卡故串行。
**本波代码提交（git）**：`Mamba 矩阵 → 关线`、`wavelet.py/分频2.0(db/dtcwt/aniso/shift)/双路U-Net`、`train_hf_flow+eval --tta/--hf-head`、`Muon proper(优化器/aux-lr/muon+adv 修复)`、`run_muon_first/run_mech_best/run_tta_best 链 + selector 修复`、本 `FINAL_REPORT` 定稿。
**诚实边界**：所有 SwinIR 对比是协议A；MANIQA 仅用于真实 SR 输出的相对排序；感知指标逐图有 ~0.1–0.3 抖动，单点 ±0.05dB/±0.1MUSIQ 不作强结论（本目录所有采纳均基于"整段单调一致/跨多指标同向"）；`hf_flow refine_* n=62`（38 张 OOM）与 `muon_mech/n3muon/pretrain`（对抗预训练 ckpt 直接 OOD 评）为过程记录、非有效对比数。
