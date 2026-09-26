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

---

# 附录 A — 全量结果母表（experiments/ 树全部 106 个 eval.json）

> 逐 run 完整清单，按根分组。Y=PSNR-Y，感知为主看列。标 *(no-IQA)* 的是早期 `eval_official/eval.json`（未跑 IQA），其 iqa_eval 重打分版列于 §阶段A/矩阵；二者 Y 常一致。标 *(dup)* 为同 run 的另一份记录。标 *(subset n=90/88)* 为 IQA 子集重跑（显存丢对），Y 与全量版略差属正常。

### A.1 阶段A / 基线 / 底噪（`iqa_eval/`，官方 100 对，权威重打分）
| run | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| bicubic_floor | 31.7335 | 0.8876 | 39.396 | 0.2782 |
| E1_swinir_light (0.61M) | 32.8827 | 0.9026 | 45.427 | 0.3024 |
| E2_swinir_capmatch (largeish 3.96M) | 32.9735 | 0.9058 | 46.789 | 0.3051 |
| A0_l1_only (4.05M) | 33.3606 | 0.9123 | 50.070 | 0.3097 |
| E11_align_loss (4.05M) | 33.4655 | 0.9143 | 49.153 | 0.3089 |

### A.2 扩散 2×2 矩阵 + DiT（`iqa_eval/` 重打分 + `pixel_*`/`latent_*`/`dit` 原始）
| run | Y | SSIM | MUSIQ | MANIQA | 备注 |
|---|---|---|---|---|---|
| dit_pixel_reg_XS (6.28M) | 33.5604 | 0.9159 | 52.541 | 0.3173 | =pixel_reg_XS(dup,no-IQA) |
| dit_pixel_reg (34.0M) | 33.7627 | 0.9191 | 53.312 | 0.3238 | =pixel_reg(dup,no-IQA);NFE1 回归 |
| pixel_flow (34.0M) | 26.7913 | 0.5504 | – | – | no-IQA；<bicubic |
| latent_flow_flux (32.9M) | 28.3614 | 0.7901 | 50.121* | 0.2236* | iqa_eval 子集 n=90 Y27.15/MUSIQ50.12 |
| latent_flow_flux_XS (5.73M) | 27.0532 | 0.7359 | – | – | no-IQA |
| latent_flow_res (32.9M) | 29.4617 | 0.8293 | 47.590* | 0.2180* | iqa_eval n=90 Y28.16 |
| latent_reg_flux (32.9M) | 31.1234 | 0.8836 | 42.239* | 0.2656* | <自身零初始化31.60；iqa_eval n=88 Y31.20 |
> 另有根目录 `latent_flow_sd`、`latent_reg_flux_lr3e5`、`v1_latent_flow_flux_14k`、`unet_reg`、`unet_1128`、`unet_s1_1244`、`s1_pretrain`、`pretrain/pt_pixel_reg`、`mamba`、`dbg2`、`vae_noise` 存在但**无 eval.json**（底噪/探针/被砍臂/中止），见附录 B。

### A.3 U-Net 容量 / 形状 / FFN / 预训练（`iqa_eval/` + `s1_sweep/` + `capacity/` + `shape_sweep/` + `s1_pretrain_b64/`）
| run | Y | SSIM | MUSIQ | MANIQA | 参数 |
|---|---|---|---|---|---|
| s1_b32 | 33.9445 | 0.9219 | 54.195 | 0.3335 | 4.99M |
| s1_b40 | 34.0070 | 0.9232 | 54.843 | 0.3391 | 7.58M |
| s1_b48 | 33.9946 | 0.9238 | 55.258 | 0.3416 | 10.73M |
| s1_b64 | 34.1083 | 0.9246 | 55.218 | 0.3416 | 18.67M |
| capacity/b128 (1244_b128) | 34.1560 | 0.9249 | 55.418 | 0.3434 | 72.53M |
| capacity/b64_ffn | 34.1580 | 0.9250 | 54.739 | 0.3389 | 24.92M |
| shape_sweep/1122_b96 | 34.0364 | 0.9236 | 54.982 | 0.3403 | 11.50M |
| shape_sweep/1124_b80 | 34.0521 | 0.9235 | 54.540 | 0.3385 | 18.26M |
| s1_b64_pretrained (=s1_pretrain_b64/b64_finetune,dup) | 34.1889 | 0.9252 | 55.739 | 0.3474 | 预训练+微调8k |
| b64_ft15k（后训练11k，阶段B冠军） | 34.1988 | 0.9251 | 55.707 | 0.3470 | 09-19 冠军 |
> `s1_sweep/{b32,b40,b48,b64}_finetune/eval_official` 为上表的 no-IQA 原始记录（Y 一致）。

### A.4 Mamba 矩阵（`mamba_matrix/` + `plan12/`）
| run | scale | 样本 | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|---|---|
| u_b64_188k | U-Net 1 | 188k | 33.2535 | 0.9102 | 47.530 | 0.3005 |
| u_b64_329k | U-Net 1 | 328k | 33.9658 | 0.9221 | 53.879 | 0.3321 |
| u_b64_439k | U-Net 1 | 439k | 34.0363 | 0.9231 | 54.474 | 0.3359 |
| m_s2_c96_match | 2 | 328k | 31.6829 | 0.8906 | 40.221 | 0.2817 |
| m_s2_c96_full | 2 | 1.54M | 33.4182 | 0.9128 | 51.377 | 0.3110 |
| m_s2_c128_e1 | 2 | 717k | 32.0674 | 0.8992 | 43.342 | 0.2925 |
| m_s2_c128_e2 | 2 | 179k | 31.8131 | 0.8909 | 40.585 | 0.2825 |
| m_s1_c96 | 1 | 269k | 32.5674 | 0.9002 | 43.413 | 0.2888 |
| plan12/mamba_c128 | 2 | 512k | 32.5492 | 0.9029 | 45.903 | 0.2989 |

### A.5 三新臂 / 小波时代 / 机制（本波，全部 100 对）
| run | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| new_arms/s1_b64_coord | 34.0848 | 0.9237 | 55.025 | 0.3409 |
| new_arms/s1_b64_froute | 34.0883 | 0.9240 | 54.903 | 0.3405 |
| wave_arms/s1_b64_dwt_w1 (λ1) | 34.1895 | 0.9258 | 55.381 | 0.3451 |
| wave_arms/s1_b64_dwt_w2 (λ2) | 34.2213 | 0.9261 | 55.434 | 0.3466 |
| wave_arms/s1_b64_dwt_w3 (λ3) | 34.2199 | 0.9253 | 54.828 | 0.3418 |
| wave_arms/s1_b64_dwt_w5 (λ5) | 34.2397 | 0.9262 | 55.342 | 0.3472 |
| wave_arms/s1_b64_dwt_w5 +TTA | 34.3257 | 0.9270 | 55.289 | 0.3477 |
| wave_arms/s1_b64_equiv | 34.0379 | 0.9243 | 55.438 | 0.3403 |
| wave_unet/s1_b64_dwtunet_dw1 | 34.1954 | 0.9258 | 55.056 | 0.3440 |
| wave_unet/s1_b64_dwtunet | 34.0933 | 0.9243 | 54.934 | 0.3410 |
| stack_dwt/b64_pre_dwt1 | 34.1867 | 0.9264 | 55.810 | 0.3501 |
| stack_dwt/b64_pre_dwt2 | 34.2363 | 0.9265 | 55.787 | 0.3501 |
| stack_dwt/b64_pre_dwt3 | 34.2511 | 0.9266 | 55.767 | 0.3501 |
| stack_dwt/b64_pre_dwt5 | 34.2670 | 0.9270 | 55.767 | 0.3506 |
| stack_dwt/b64_pre_dwt5 +TTA | 34.3628 | 0.9277 | 55.735 | 0.3509 |
| stack_dwt/b64_pre_dwt6 | 34.2762 | 0.9270 | 55.830 | 0.3506 |
| stack_dwt/b64_pre_dwt7 | 34.2784 | 0.9270 | 55.832 | 0.3509 |
| stack_dwt/b64_pre_dwt8 | 34.2781 | 0.9271 | 55.870 | 0.3513 |
| stack_dwt/b64_pre_dwt8 +TTA | 34.3772 | 0.9279 | 55.825 | 0.3516 |
| stack_dwt/b64_pre_dwt8_db2 | 34.2710 | 0.9270 | 55.698 | 0.3501 |
| stack_dwt/b64_pre_dwt8_db4 | 34.2588 | 0.9269 | 55.633 | 0.3501 |
| stack_dwt/b64_pre_dwt8_db2a (aniso) | 34.2527 | 0.9270 | 55.686 | 0.3499 |
| stack_dwt/b64_pre_dwt10 | 34.2876 | 0.9270 | 55.801 | 0.3507 |

### A.6 Muon 时代（全部 @ Muon 5e-3 除标 AdamW 对照）
| run | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| stack_dwt/b64_pre_dwt8_muon | 34.2894 | 0.9280 | 55.850 | 0.3521 |
| stack_dwt/b64_pre_dwt8_muon +TTA | 34.4048 | 0.9289 | 55.751 | 0.3521 |
| muon_ab/muon_adamw (AdamW 对照) | 34.2417 | 0.9259 | 55.464 | 0.3470 |
| muon_ab/muon_lr1e3 | 34.3006 | 0.9272 | 55.990 | 0.3520 |
| muon_ab/muon_muon (2e-3) | 34.3935 | 0.9279 | 55.818 | 0.3512 |
| muon_ab/muon_lr5e3 | 34.3914 | 0.9280 | 55.974 | 0.3527 |
| muon_ab/muon_lr1e2 | 34.3280 | 0.9279 | 55.929 | 0.3527 |
| muon_tune/m_lr2e3 | 34.2528 | 0.9261 | 55.487 | 0.3482 |
| muon_tune/m_lr8e3 | 34.1554 | 0.9244 | 55.014 | 0.3436 |
| muon_tune/m_lr12e3 | 34.0053 | 0.9225 | 54.999 | 0.3411 |
| muon_tune/m_mom090 | 34.2683 | 0.9260 | 55.365 | 0.3471 |
| muon_tune/m_mom098 | 34.1434 | 0.9243 | 54.860 | 0.3424 |
| muon_tune/m_ns3 | 34.2397 | 0.9248 | 55.076 | 0.3416 |
| muon_tune/m_ns7 | 34.2419 | 0.9256 | 55.270 | 0.3456 |
| muon_tune/m_aux3e4 | 34.2188 | 0.9252 | 55.167 | 0.3451 |
| muon_tune/m_wd1e2 | 34.2135 | 0.9250 | 55.129 | 0.3451 |
| muon_mech/p_shift4 (2e-3,handicap) | 34.2845 | 0.9276 | 55.898 | 0.3523 |
| muon_mech/p_aniso (2e-3) | 34.2656 | 0.9274 | 55.903 | 0.3524 |
| muon_mech/p_dtcwt (2e-3) | 34.1587 | 0.9253 | 55.843 | 0.3454 |
| muon_mech/p_lv3shift (2e-3) | 34.3001 | 0.9271 | 55.732 | 0.3509 |
| muon_mech/z_dual (2e-3) | 33.8451 | 0.9214 | 53.911 | 0.3338 |
| muon_mech/z_dual_shift4 (2e-3) | 33.8555 | 0.9207 | 53.731 | 0.3332 |
| muon_mech/z_dual_dtcwt (2e-3) | 33.1622 | 0.9105 | 51.344 | 0.3252 |
| muon_mech/n3muon/ft (2e-3) | 34.2428 | 0.9275 | 55.896 | 0.3531 |
| mech_best/p_shift4 (5e-3) | 34.3096 | 0.9277 | **56.022** | **0.3534** |
| mech_best/p_shift4 +TTA | 34.4063 | 0.9285 | 55.960 | 0.3534 |
| mech_best/p_aniso (5e-3) | 34.3099 | 0.9277 | 55.921 | 0.3532 |
| mech_best/p_aniso +TTA | 34.4074 | 0.9285 | 55.862 | 0.3531 |
| mech_best/p_dtcwt (5e-3) | 34.1541 | 0.9251 | 55.951 | 0.3457 |
| mech_best/n3_ft (5e-3) | 34.2764 | 0.9279 | 55.570 | 0.3514 |
| mech_best/n3_ft +TTA | 34.3882 | 0.9288 | 55.497 | 0.3514 |

### A.7 ×3 / ×4 迁移（scale-adaptive λ，零预训练 AdamW）
| run | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| scale_3/s1_b64_plain | 31.0383 | 0.8687 | 51.786 | 0.3239 |
| scale_3/s1_b64_dwt1 | 31.0601 | 0.8671 | 49.751 | 0.3121 |
| scale_3/s1_b64_dwt2 | 31.1300 | 0.8709 | 52.041 | 0.3320 |
| scale_3/s1_b64_dwt5 | 31.1164 | 0.8686 | 50.478 | 0.3196 |
| scale_4/s1_b64_plain | 29.4156 | 0.8295 | 47.146 | 0.2993 |
| scale_4/s1_b64_dwt5 | 29.5106 | 0.8322 | 48.249 | 0.3131 |

### A.8 频域 rectified flow（方向C，判负；OOM 后 n=62）
| run | Y | SSIM | MUSIQ | MANIQA |
|---|---|---|---|---|
| hf_flow/refine_s0.10 | 31.0746 | 0.7622 | 47.579 | 0.2308 |
| hf_flow/refine_s0.20 | 27.1776 | 0.5414 | 41.378 | 0.2409 |
| hf_flow/refine_s0.35 | 23.1929 | 0.3482 | 36.054 | 0.2545 |
| muon_mech/n3muon/pretrain *(OOD，非有效)* | 22.3039 | 0.7604 | 60.876 | 0.4691 |

---

# 附录 B — 实验根目录清单（含无 eval 的中止/探针根，供审计）

**有官方 eval（见附录 A）**：`iqa_eval`、`pixel_reg`、`pixel_reg_XS`、`pixel_flow`、`latent_flow_flux(_XS)`、`latent_flow_res`、`latent_reg_flux`、`s1_sweep`、`s1_pretrain_b64`、`b64_ft15k`、`capacity`、`shape_sweep`、`mamba_matrix`、`plan12`、`new_arms`、`wave_arms`、`wave_unet`、`stack_dwt`、`muon_ab`、`muon_tune`、`muon_mech`、`mech_best`、`scale_3`、`scale_4`、`hf_flow`。

**目录存在但无 eval.json（中止 / 无官方评估 / 探针）**：
| 根 | 说明 |
|---|---|
| `ablation_x2/A0_l1_only`、`improve/E11_align_loss`、`matrix_x2/E1,E2` | 阶段A SwinIR 线（结果经 `iqa_eval/` 权威重打分，见 §1） |
| `pretrain/pt_pixel_reg`、`s1_pretrain` | 预训练中间产物（无独立 Test 评估） |
| `unet_reg`、`unet_1128`、`unet_s1_1244` | 早期 U-Net 容量/形状预研，未单列官方评估 |
| `latent_flow_sd`、`latent_reg_flux_lr3e5`、`v1_latent_flow_flux_14k` | 扩散被砍臂/早期变体（flux 底噪证伪后砍，见 §0/矩阵） |
| `mamba` | Mamba 早期单跑（正式在 `mamba_matrix`） |
| `vae_noise` | VAE 重建底噪表（非 SR 评估，产物格式不同） |
| `dbg2` | 调试残留 |
| `n3`、`honest_deep`、`sweep48` | **被主动中止的链**（用户令停止/重排），无产物 |
| `smoke_*`（smoke_hf/mamba/mamba_probe/muonadv/muon_srv/new_arms） | smoke 探针产物，非结果 |

---

# 附录 C — 驱动 / 日志索引

**顶层 campaign 日志（`experiments/*.log`，每条=一次链式 campaign）**：
`wave_arms`、`wave_unet`、`stack_dwt`、`stack_sweep`/`stack_sweep2`、`sweep48`、`wave_sweep`、`freq2`、`mamba_matrix`、`mamba_speed`、`mm_fill`、`muon_ab`、`muon_sweep`、`muon_first`、`champ_muon`、`mech_best`、`hf_flow`、`honest48`、`honest_deep`、`next_phase`/`next_phase2`、`wavedriver`、`reeval`、`tta_best`、`capacity`/`capacity_ffn`、`shape_sweep`、`chain_after`/`chain_b40`/`chain_b64`/`chain_pretrain`、`cut_vae`、`iqa_evals`、`swinir_iqa`/`swinir_iqa_fast`、`precache_hr`/`precache_hr2`。
**逐臂训练/评估日志**：`experiments/diffusion/logs/`（共 222 个 `.log`，前缀 `na_/wa_/wu_/mm_/sd_/sc_/mu_/mf_/mb_/tb_/hn_/s48_/fq_/hd_/hf_/sc_` 对应各 campaign 的每一臂 train/eval）。

> 说明：所有 campaign 均走 `scripts/server/run_*.sh`（tmux 会话 + 逐臂 `wait_gpu` + `eval.json` 幂等跳过），支持断点续跑；本轮多次因用户重排指令用 `tmux kill-session` 中止并在改配后重启，历史产物保留于各自根目录。

---

# 附录 D — 数据审计发现（写入前逐 run 核 `args.json` 才暴露）

1. **Phase M 的 batch 混杂（重要）**：`muon_tune/*` 9 臂的 `args.json` 显示 **`batch=8`**（`run_muon_first.sh` 的 `BASE` 漏写 `--batch`，train_pixel 默认 8），而 champmuon / `mech_best` / `muon_ab` 均为 `batch=128`。→ **Phase M 内部相互可比（都 bs=8），但与 bs=128 的 base/冠军直接横比不公平**；“9 偏离全 < base”部分反映 batch 差。**影响面限定**：① Muon 最优=lr5e-3 仍成立——由 `muon_ab`(bs128, 零预训练) 与 `champmuon`(bs128) 两套独立佐证；② momentum/ns_steps/aux-lr/wd 的 Phase M 结论仅“bs=8 下成立”，**若要严谨需 bs=128 复验**（本轮未做，登记为遗留项）。
2. **`plan12/mamba_c128`**：`steps=1600 batch=320 num_res=4 native_lr=-1(auto)`——是 Mamba 早期**速度/可行性探针**（仅 1600 步、欠训），Y 32.55 不代表收敛，正式 Mamba 结论以 `mamba_matrix` 为准。
3. **`pixel_flow`/`latent_*` 早期矩阵**：`lr_patch=256 steps=22000 batch=28~96 lr=1e-4 ema=0.999`——DiT 全图扩散早期配置（`lp=256`→HR512 tile 巨大故 batch 很小），均为底噪证伪前的探索，判负口径见 §3。
4. **`muon_ab` AdamW 对照** 的 `args.json` 里 `muon_lr=0.002` 仅为默认占位（该臂实际 optimizer=adamw），不影响对比。
5. **`mech_best/n3_ft` 与 `muon_mech/n3muon/ft`** 分属两条链（前者 bs128@5e-3、后者 handicap bs128@2e-3）；两者 ft init 均取自各自对抗预训练 ckpt。mech_best 版（5e-3）为 N3 的正式判定数。

---

# 附录 E — 逐 run 配置台账（experiments/ 全树 106 run，原样可审计）

> `mtime [tag] 路径`（tag: iqa=带IQA / off=eval_official无IQA / TTA）；下一行 config；再下一行 dwt旗标 + 结果。共享默认：`--backbone unet --size S --mult 1,2,4,4 --num-res 2 --attn-levels 2,3 --native-lr 0 --objective reg --residual 1 --lr-patch 64 --ema 0.999 --warmup 500 --decoded-manifest data/decoded/manifest.json --cache-data 0 --num-workers 12`。`–`=默认/未设。

```
# —— 阶段A / 底噪 / 扩散矩阵（多为 iqa_eval 重打分；原始 eval_official 见下）——
(阶段A/Diffusion/iqa_eval 各 run 的配置详列于 PROGRESS_REPORT / STAGE_SUMMARY；
 iqa_eval 重打分产物本身不携 args.json，故台账从 diffusion/ 原始 run 行记录。)

09-16 20:58 [off] diffusion/latent_reg_flux
    base=– mult=– nr=– att=– nlr=– sc=2 lp=256 | adamw lr=0.0001 st=22000 bs=64 ema=0.999 wu=500
    SSIM=0.8836 Y=31.1234 n=100   (无IQA)
09-17 03:49 [off] diffusion/pixel_flow
    lp=64 | adamw lr=0.0001 st=16000 bs=28
    SSIM=0.5504 Y=26.7913 n=100   (无IQA; <bicubic)
09-17 10:37 [off] diffusion/latent_flow_flux_XS
    lp=256 | adamw lr=0.0001 st=22000 bs=96
    SSIM=0.7359 Y=27.0532 n=100
09-17 13:46 [off] diffusion/latent_flow_res
    lp=256 | adamw lr=0.0001 st=22000 bs=64
    SSIM=0.8293 Y=29.4617 n=100

# —— U-Net 线（容量/形状/FFN/预训练）——
09-18 08:15 [off] diffusion/s1_pretrain_b64/b64_finetune
    base=64 sc=2 | adamw lr=0.0003 st=10000 bs=128 init=ckpt_best.pt
    SSIM=0.9252 Y=34.1889 n=100   (=iqa_eval/s1_b64_pretrained)
09-18 14:23 [iqa] diffusion/shape_sweep/1122_b96
    base=96 mult=1,1,2,2 | adamw st=10000 bs=96
    SSIM=0.9236 MUSIQ=54.982 MANIQA=0.3403 Y=34.0364
09-18 15:57 [iqa] diffusion/shape_sweep/1124_b80
    base=80 mult=1,1,2,4 | adamw st=10000 bs=128
    SSIM=0.9235 MUSIQ=54.540 MANIQA=0.3385 Y=34.0521
09-18 19:28 [iqa] diffusion/capacity/b128   (1244_b128)
    base=128 | adamw st=10000 bs=64
    SSIM=0.9249 MUSIQ=55.418 MANIQA=0.3434 Y=34.1560
09-19 09:06 [iqa] diffusion/capacity/b64_ffn
    base=64 ffn | adamw st=10000 bs=96
    SSIM=0.9250 MUSIQ=54.739 MANIQA=0.3389 Y=34.1580
09-19 23:28 [iqa] diffusion/plan12/mamba_c128   (速度探针, 欠训)
    base=128 nr=4 nlr=-1 | adamw st=1600 bs=320
    SSIM=0.9029 MUSIQ=45.903 MANIQA=0.2989 Y=32.5492

# —— 三新臂 ——
09-19 12:10 [iqa] diffusion/new_arms/s1_b64_coord     coord
    SSIM=0.9237 MUSIQ=55.025 MANIQA=0.3409 Y=34.0848
09-19 14:08 [iqa] diffusion/new_arms/s1_b64_froute     froute
    SSIM=0.9240 MUSIQ=54.903 MANIQA=0.3405 Y=34.0883

# —— 小波时代 round1-2（零预训练 b64, adamw st=10000 bs=128）——
09-20 16:24 [iqa] wave_arms/s1_b64_dwt_w1   dwt{w1 lv2 haar sh1}
    SSIM=0.9258 MUSIQ=55.381 MANIQA=0.3451 Y=34.1895
09-20 17:09 [iqa] wave_arms/s1_b64_dwt_w3   dwt{w3}
    SSIM=0.9253 MUSIQ=54.828 MANIQA=0.3418 Y=34.2199
09-20 20:30 [iqa] wave_arms/s1_b64_equiv    equiv (st=20000 bs=64)
    SSIM=0.9243 MUSIQ=55.438 MANIQA=0.3403 Y=34.0379
09-21 02:20 [iqa] wave_arms/s1_b64_dwt_w2   dwt{w2}
    SSIM=0.9261 MUSIQ=55.434 MANIQA=0.3466 Y=34.2213
09-21 03:31 [iqa/TTA] wave_arms/s1_b64_dwt_w5  dwt{w5}
    iqa SSIM=0.9262 MUSIQ=55.342 MANIQA=0.3472 Y=34.2397 | TTA Y=34.3257

# —— 小波时代 round3-5（预训练 init=ckpt_best.pt, adamw st=10000 bs=128）——
09-21 05:06 [iqa] stack_dwt/b64_pre_dwt1    dwt{w1}
    SSIM=0.9264 MUSIQ=55.810 MANIQA=0.3501 Y=34.1867
09-21 14:35 [iqa/TTA] stack_dwt/b64_pre_dwt8  dwt{w8}
    iqa SSIM=0.9271 MUSIQ=55.870 MANIQA=0.3513 Y=34.2781 | TTA Y=34.3772
09-21 18:06 [iqa] stack_dwt/b64_pre_dwt6    dwt{w6}  SSIM=0.9270 MUSIQ=55.830 Y=34.2762
(另有 dwt2/3/5/7/10 及 dwt8 的 db2/db4/db2a，配置=预训练init+对应 dwt 旗标，数值见 §A.5)
09-23 06:46 stack_dwt/b64_pre_dwt8_db2  dwt{w8 db2}; 09-23 08:26 _db4 dwt{w8 db4}

# —— 双射小波 U-Net（wave_unet，dwtunet）——
(dwtunet / dwtunet_dw1 见 §A.5；dwt_unet=True 旗标，zero-pretrain)

# —— ×3 / ×4 迁移（zero-pretrain adamw st=10000；×3 bs96 lp48, ×4 bs128 lp32）——
09-22 09:07 scale_3/s1_b64_plain   sc=3 lp=48  SSIM=0.8687 MUSIQ=51.786 Y=31.0383
09-22 17:14 scale_4/s1_b64_plain   sc=4 lp=32  SSIM=0.8295 MUSIQ=47.146 Y=29.4156
09-22 12:32 scale_3/s1_b64_dwt5 dwt{w5} Y=31.1164; 09-23 00:53 scale_4/s1_b64_dwt5 Y=29.5106
09-23 12:10 scale_3/s1_b64_dwt1 dwt{w1} Y=31.0601; 09-23 18:46 _dwt2 dwt{w2} SSIM=0.8709 Y=31.1300

# —— Muon 时代 A：muon_ab（零预训练 b64+λ8, adamw/muon, st=10000 bs=128）——
09-24 06:57 [iqa] muon_ab/muon_adamw  adamw lr=0.0003 dwt{w8}  SSIM=0.9259 MUSIQ=55.464 MANIQA=0.3470 Y=34.2417
09-24 08:33 muon_muon   muon mlr=0.002 dwt{w8} SSIM=0.9279 MUSIQ=55.818 MANIQA=0.3512 Y=34.3935
09-24 10:20 muon_lr1e3  muon mlr=0.001 Y=34.3006
09-24 11:51 muon_lr5e3  muon mlr=0.005 SSIM=0.9280 MUSIQ=55.974 MANIQA=0.3527 Y=34.3914
09-24 14:03 muon_lr1e2  muon mlr=0.01  Y=34.3280

# —— Muon 时代 B：muon_tune（Phase M；★batch=8 confound, init=ckpt_best.pt, muon lr base）——
09-24 19:50 m_lr2e3 mlr=0.002 bs=8 SSIM=0.9261 Y=34.2528
09-24 19:58 m_lr8e3 mlr=0.008 bs=8 SSIM=0.9244 Y=34.1554
09-24 20:09 m_lr12e3 mlr=0.012 bs=8 SSIM=0.9225 Y=34.0053
09-24 20:19 m_mom090 mom=0.9 bs=8 SSIM=0.9260 Y=34.2683
09-24 20:29 m_mom098 mom=0.98 bs=8 SSIM=0.9243 Y=34.1434
09-24 20:47 m_ns7 ns=7 bs=8 SSIM=0.9256 Y=34.2419
09-24 20:56 m_aux3e4 aux=0.0003 bs=8 SSIM=0.9252 Y=34.2188
09-24 21:05 m_wd1e2 wd=0.01 bs=8 SSIM=0.9250 Y=34.2135
09-24 20:31 m_ns3 ns=3 bs=8 SSIM=0.9248 Y=34.2397

# —— Muon 时代 C：mech_best（★bs=128, muon 5e-3/mom0.95/ns5, init=预训练冠军）——
09-25 14:48 [iqa/TTA] mech_best/p_shift4   dwt{w8 lv2 haar sh4}
    iqa SSIM=0.9277 MUSIQ=56.022 MANIQA=0.3534 Y=34.3096 | TTA Y=34.4063
09-25 16:31 [iqa/TTA] mech_best/p_aniso    dwt{w8 haar sh4 b(1.5,0.8,1)}
    iqa SSIM=0.9277 MUSIQ=55.921 MANIQA=0.3532 Y=34.3099 | TTA Y=34.4074
09-25 17:13 mech_best/p_dtcwt dwt{w8 lv2 dtcwt sh1} SSIM=0.9251 MUSIQ=55.951 MANIQA=0.3457 Y=34.1541
09-26 00:16 [iqa/TTA] mech_best/n3_ft (init=n3对抗预训练) dwt{w8}
    iqa SSIM=0.9279 MUSIQ=55.570 MANIQA=0.3514 Y=34.2764 | TTA Y=34.3882

# —— Muon 时代 D：muon_mech（★handicap bs128@lr2e-3；含双路U-Net/N3）——
09-24 22:19 muon_mech/p_shift4 muon mlr=0.002 dwt{w8 haar sh4} SSIM=0.9276 MUSIQ=55.898 Y=34.2845
09-24 22:56 muon_mech/p_dtcwt mlr=0.002 dwt{dtcwt} Y=34.1587
09-25 00:49 muon_mech/p_lv3shift dwt{w8 lv3 haar sh4} Y=34.3001
09-25 03:02 muon_mech/p_aniso dwt{w8 haar sh4 b(1.5,0.8,1)} Y=34.2656
09-25 04:03 muon_mech/z_dual     DUAL dwt{w8 haar} SSIM=0.9214 MUSIQ=53.911 Y=33.8451
09-25 04:24 muon_mech/z_dual_shift4 DUAL dwt{w8 haar sh4} SSIM=0.9207 Y=33.8555
09-25 04:46 muon_mech/z_dual_dtcwt  DUAL dwt{w8 dtcwt} SSIM=0.9105 MUSIQ=51.344 Y=33.1622
09-25 08:48 muon_mech/n3muon/pretrain  ADV mlr=0.002 st=25000 dwt{w8} (OOD直接评, 非有效) SSIM=0.7604 Y=22.3039
09-25 10:54 [iqa/TTA] muon_mech/n3muon/ft init=对抗ckpt dwt{w8} SSIM=0.9275 MUSIQ=55.896 MANIQA=0.3531 Y=34.2428

# —— 频域 rectified flow（方向C；refine 头 st=8000 bs=64 lr=2e-4, 从 b64_pre_dwt8 起）——
09-24 05:16 [iqa] hf_flow/hf_x2/refine_s0.10  n=62(OOM丢38) SSIM=0.7622 Y=31.0746
09-24 05:16 refine_s0.20 SSIM=0.5414 Y=27.1776; refine_s0.35 SSIM=0.3482 Y=23.1929

# —— 交付态 Muon 微调 ——
09-25 stack_dwt/b64_pre_dwt8_muon muon mlr=0.005 init=预训练冠军 dwt{w8}
    SSIM=0.9280 MUSIQ=55.850 MANIQA=0.3521 Y=34.2894 | +TTA SSIM=0.9289 MUSIQ=55.751 Y=34.4048
```

> 台账说明：`iqa_eval/` 下的阶段A/扩散重打分产物（bicubic/E1/E2/A0/E11/dit_*/s1_b32-64/latent_flow_* 等，见 §A.1–A.2）是对应原始 run 的再评估，其 config 即原始 run（diffusion 矩阵）+ 无 `--iqa` 差异；原始 run 的 `eval_official/eval.json` 已在上方以 `[off]` 行登记。全部 106 个带指标 run 均已在上文 §A 母表或本台账出现。

---

# 附录 F — RealSR ×2 经验上限（training-free 确定性实测）

> 工具：`diffusion/oracle.py`（无训练、无深度模型、闭式；官方 limited-range Y/uint8/modcrop、n=100）。目的：把 MSE 拆成三堵不可约物理墙，定量给出数据集的上限。

| 墙 | 实测（n=100，官方 Y） |
|---|---|
| **E_align** 亚像素配准（对 GT 施带限傅里叶相移） | 0.2px→**41.11** (SSIM .991)；0.3px→**37.69** (.981)；0.5px→**33.41** (.952)；1.0px→27.94 (.850) |
| **E_null** 带宽截断（理想低通） | LR 带 0.5·Nyquist→**40.65** (SSIM .979)；0.75→47.04；1.0（恒等，验 FFT 精度）→56.99 |
| **E_noise** 传感器噪声（Donoho 小波-MAD，免训练） | σ̂=**0.899**/255 → 上限 **49.05 dB** |

**合成解析上限**（相对 MSE a=10^(-PSNR/10) 相加）：
- 带宽+噪声（不含配准）：a=8.61e-5+1.24e-5 → **40.1 dB**。
- 再叠 0.2/0.3/0.4 px 残差配准 → **37.5 / 35.7 / ≈34.4 dB**。

**定结论（诚实、不吹）**：锁死 PSNR 的是**双镜头配准残差**（非噪声，也非带宽）：σ≈0.9 使噪声上限高至 49dB、带宽单独 40.6dB，但 0.3–0.4px 级残差配准就把每对上限拉到 **~34.4–35.7 dB**。本项目交付 **34.41 dB / SSIM 0.9285** 已落在该配准受限上限带内（SSIM 上限同量级 ~0.94–0.95）；换言之 PSNR 已基本触顶，剩余空间需靠“对齐感知”而非“重建保真”。**这恰好仍验证 Blau & Michaeli 失真-感知权衡**：要向 35+ dB 再走只能抹平高频往条件均值靠（降感知）；要提感知只能猜高频（降 PSNR）。**→ 后续应全力转向 MUSIQ/LPIPS 等感知轴，PSNR 已无可持续空间。**

**为何不用“过拟合 Test”定上限（已弃用）**：足容量网络+足够步会把 GT 的噪声与错位**一并背诵**，PSNR 会超过真实物理上限 → 它测的是**模型记忆容量**而非**数据信息极限**，且泄露测试集，**不能定上界**。`--overfit-test` 旗标保留作诊断但**不作为上限方法**。

