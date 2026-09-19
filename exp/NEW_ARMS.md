# 三个新方案：实现与 A/B 设计（2026-09-19）

> 背景：`STAGE_SUMMARY.md` 的结论是「架构 ≫ 容量 > 数据 > 步数」，且三条剩余瓶颈分别是
> 卷积平移不变性（真实镜头退化是非平稳的）、门控 FFN 的无差别放大（平滑区噪声）、
> 390 对数据的有限支撑集（2460 epoch）。三个方案各对准一条。
> 全部实现已过本地 smoke（`python -m diffusion.smoke_new_arms`）。

---

## 方案一：INR / 坐标注入（打破平移不变性）

**机理**：stride-1 U-Net 共享一组 W，中心锐、边缘像散/渐晕的真实镜头只能学"平均核"。
把 HR crop 在**整幅图坐标系**中的绝对坐标 c=(u,v)∈[−1,1]² 作为 2 个额外输入通道拼进模型
（CoordConv 式输入注入），网络即获得按位置调制的逆退化能力 W(c)。

**实现**（v1 是输入注入，非完整 LIIF-MLP 头）：
- `data.py::RealSRCropDataset(coord=True)`：按 crop 在整幅 HR 图中的位置生成 (2,128,128) 坐标图，
  随图像做同样的 flip/rot90（坐标标签跟着内容走）；
- `unet.py::UNet(coord_channels=2)`：stem 输入 3+2 通道，**coord 通道的 stem 权重零初始化**
  → step 0 与无 coord 模型完全等价（smoke 已验证），位置通路要自己"挣"影响力；
- `train_pixel.py --coord`；仅 unet+reg（flow/DiT 不支持，启动即拒绝）；
- `eval_official.py`：分块评估时对整幅 padded 图建坐标场，每个 tile 切自己的子区间
  —— tile 看到的坐标与训练时完全一致（smoke：零初始化模型过 tiled eval 逐像素等于 bicubic）。

**局限（明说）**：BSRGANDataset 尚无坐标（`--coord + --pretrain-root` 被禁止）；
预训练要带坐标得后续给 BSRGAN 数据加同款坐标场。

## 方案二：频域自适应软路由

**机理**：FFN 的 `x1⊗x2` 无差别展开同时放大高频纹理与平滑区底噪（官方判定 ≈打平）。
改为**按像素凸组合两个专家**：
`Δ = α·F_texture(x) + (1−α)·F_smooth(x)`，
`α = sigmoid(Conv1×1(Sobel 能量))`（固定 Sobel，作用于块输入的通道均值，
保留水平/垂直两路方向能量——正是像散信号）。

**实现**（`unet.py::FreqRouter + SmoothBranch`，`train_pixel.py --freq-route`）：
- F_texture = 现有 3×3 卷积路径（零初始化不变）；F_smooth = 1×1 瓶颈(c/4)→depthwise 3×3→1×1
  （末层零初始化），参数约为半个 3×3 卷积——"低秩/小核"的操作化替身；
- 连续标量图路由，无 token-MoE 撕裂；step 0 仍是恒等映射；
- b64 参数 18.67M → **≈19.6M（+5%）**，FFN 是 +33%——容量效率完全不同。

## 方案三：对抗性退化挖掘（min-max）

**机理**：390 对 × 2460 epoch = 经验分布坍缩成离散点，加参数/步数/固定退化预训练全部饱和
（+0.05/+0.01/+0.09）。把 BSRGAN 的"固定管线采样"换成鲁棒优化
`min_θ max_φ E_x[L(F_θ(G_φ(x)), x)]`——G_φ 沿当前 SR 网络的软肋方向在**连续退化流形**上主动采样。

**实现**（`adversarial.py`，`train_pixel.py --adv-deg`，替代 `--pretrain-root` 下的 BSRGAN）：
- **约束集 Φ**（防"输出纯黑"式作弊）：模糊 = 18 个物理 PSF（6 各向同性 + 12 各向异性高斯，
  σ 覆盖 BSRGAN×2 口径并留有更难的上限）上的 **softmax 凸组合**，经 stride-2 depthwise 卷积
  一步完成"模糊+降采样"（k=15 奇数核 → 无半像素相位偏移）；噪声 σ ∈ [0, 0.12] 可微参数化；
- G_φ = 小 CNN(57.6K 参数) 看图出参数——**按内容攻击**（平滑区上噪声、纹理区上毁它频带的核）；
- min-max 调度：`--adv-inner 0`（默认）= 同步 GDA，一次反传、φ 用翻转梯度上升，**零额外前向**；
  `>0` = 真交替（每步多 N 次冻结 θ 的前向）；
- 预训练 val 用预训练域留出集（DIV2K_valid 100 张，零重叠，eps 按 idx 固定）驱动 ckpt_best
  —— 沿用已修正的口径，不用 RealSR val 挑预训练 ckpt；
- JPEG/随机 resize 不可微，v1 不进对抗集（可后续作为固定的附加随机算子）。

**对照公平性**：样本预算与 BSRGAN 预训练完全一致（25000 步 × batch 128 = 3.2M 样本），
之后同配方 RealSR 微调 → 官方评估。BSRGAN 那条只换 +0.09 dB —— 这就是要打败的数。

---

## A/B 矩阵（服务器 4090，锚点 = s1_b64 零预训练）

**锚点：Y 34.1083 / SSIM 0.9246 / MUSIQ 55.218 / MANIQA 0.3416**
（b64 + 预训练+后训练上限参考：Y 34.1988）

| 臂 | 配方（其余与锚点逐字相同） | 预期时长 | 判定 |
|---|---|---|---|
| N1 `s1_b64_coord` | + `--coord` | ~1.6h | 官方 100 对 Y vs 34.1083；另看边缘/中心 PSNR 分解是否分化（坐标生效的直接证据） |
| N2 `s1_b64_froute` | + `--freq-route` | ~1.6h | 同上；+MUSIQ/MANIQA 是否改善（路由的次要卖点是平滑区更干净） |
| N3 `adv_pretrain` → `s1_b64_adv_ft` | 25000×128 对抗预训练 → RealSR 微调 10000×128 | ~6h | **目标 > 34.1988（打败 BSRGAN 预训练线）**；同时看对抗域 val 曲线与 kernel_report（φ 用了流形哪一块） |

启动：`tmux new -d -s newarms 'bash scripts/server/run_new_arms.sh'`（幂等，出过 eval.json 的臂自动跳过）。
日志：`experiments/diffusion/logs/na_*.log`；结果：`experiments/diffusion/new_arms/<臂>/eval_iqa/eval.json`。

**判定纪律**（吸取 FFN 教训）：16 对 val 只看趋势；定生死必须用官方 100 对 + IQA；
±0.05 dB 以内按打平处理。

## smoke 记录（本地 4070，2026-09-19）

```
freq-route: identity-at-init OK, backward OK；b32 +0.10M（4.99→5.09M）
coord: 零初始化切片 OK，init 等价 OK（换坐标输出不变），反传 OK
adversary: 18 核、57.6K 参数；确定性（固定 eps）；min-max 梯度 θ/φ 双通；
           kernel_report {top: 各向异性核, entropy 2.89, mean_sigma 0.058}
tiled eval + coord: 零初始化模型逐像素复现 bicubic（|diff|=0）
dataset coord: 形状/范围 OK，8/8 crop 绝对位置互不相同
2-step 训练：plain / --coord / --freq-route 全部通过
SMOKE PASS
```
