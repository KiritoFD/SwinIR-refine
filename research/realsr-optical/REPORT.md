# RealSR 光学/逆问题本质：什么真正有效

**Date:** 2026-09-13 · **Mode:** standard deep-research · Findings: `research/realsr-optical/findings/F1–F4.md`

---

## 一句话

RealSR 不是「bicubic 的逆」，而是**两次真实光学采样之间的空间可变、非单射逆问题**；在已具备残差重建 + 4M 级 transformer 的前提下，**最值得做的不是再加 FFN 变体，而是：训练期显式处理配准误差（Mimicked/OF-in-loss），并用官方 Y 通道评估协议对齐文献。**

---

## 1. 任务的正向模型（F1）

\[
x=\mathcal D_L(S)+n_L,\quad y=T_\tau(\mathcal D_H(S))+n_H
\]

- \(\mathcal D_L,\mathcal D_H\)：不同对焦/焦距下的真实成像（PSF、像差、场曲、噪声），**不是**固定核 \(\downarrow(h*y)\)。
- \(T_\tau\)：配准后仍残留的非刚性错位（经验约 **0.5–3 px**，极端可到数十 px）[F2]。
- 配准后算子仍是**空间可变且非单射** → SR 是集合值逆；高频部分靠先验「补」，不是无中生有 [F1]。

**含义：** 全局平移不变卷积/注意力在物理上就错；必须估计**局部核 / 偏移**（或等价条件化），而不是只堆深度 [F1][F3]。

---

## 2. 文献里真正动过 RealSR 数字的杠杆（F2–F4）

| 杠杆 | 证据量级 | 机制 | 对本仓库 |
|------|----------|------|----------|
| **真实配准对 vs 合成 BD** | ~+0.5–1.0 dB（Y, ×2）[F3][F4] | 数据本身 | 已有 RealSR V3 |
| **Mimicked Alignment / OF-in-loss**（训练期） | SwinIR/RRDB 上 **+0.4～2.2 dB** vs 纯 L1；相对 OF-only +1.2 dB；**推理零开销** [F2] | 让 loss 比较的是对齐后的像素，而不是错位像素 | **尚未做** |
| **空间可变核 LP-KPN** | 同容量下优于直接合成 +0.1–0.2 dB；官方 V2 ×2 **33.49 / 0.917（Y）** [F3] | 显式局部滤波 | 仅有门控残差 KPN，非 LP-KPN |
| **官方评估：RGB 训练 / Y 测试** | 绝对 PSNR 与 RGB 差 **约 1–2 dB**，跨论文不可比 [F4] | 协议 | **我们一直用 RGB PSNR** |
| 测试期 KernelGAN 类核匹配 | RealSR 类核上 ~+0.5–1 dB [F4] | 每图估核 | 可作 backup |
| GAN / Real-ESRGAN 风格 | 提 MOS/NIQE，**不涨 paired Y-PSNR** [F3] | 感知 | 不作主线 |
| 再堆 CX/CoBi 作重建 loss | 在 RealSR 迁移 **−2～−3 dB** [F2] | — | 已证实 UWCL 有害 |

---

## 3. 我们实验矩阵在说什么

- 6k 步消融：patch-amp / conf-HF / EMA **≤0.03 dB**（噪声）——与「loss 项不是主矛盾」一致。
- UWCL 主损失 **−0.95 dB**——与 F2「CX 类不可作 RealSR 重建 loss」一致。
- 残差 bicubic + 长 schedule 是优化/条件化收益，**没有解决 \(T_\tau\) 与 \(\mathcal D\) 的空间可变性**。

因此下一轮应做 **物理上必需** 的干预，而不是更多谱域正则。

---

## 4. 收敛后的两条真正值得做的方法

### 主推（唯一主实验）：训练期 Mimicked Alignment / 光流对齐 loss

**想法：** 前向仍 \(\hat y=f_\theta(x)\)；损失在 **把 \(\hat y\) 与 \(y\) 拉到同一局部坐标** 后计算（或对 \(y\) 做 mimicked LR 域对齐），使 L1/高频不再被错误对应惩罚。推理图与现在完全相同 → **零部署成本** [F2]。

**为何是本质解：** 直接消掉 \(p(y|x)=\int p(y|\tau,x)p(\tau|x)d\tau\) 中的 \(\tau\) 边缘化近似误差；比「给错位像素降权」（我们试过的 conf-HF）更对症 [F1][F2]。

**实现要点（保持小、可验证）：**
1. 训练时用冻结/轻量 flow（或 grid-sample + 学习偏移）把 HR 对齐到 SR，再算 L1（+轻量 amp 可选）。
2. 偏移幅度正则，防止学成任意 warp。
3. 与现有 Mod-v2 主干、EMA、Y 评估一起；**一次** 15k 级训练 vs 同协议 L1 基线。

### 备选（若 align 实现风险高）：官方协议 + LP-KPN 头

- 评估改 **Y-PSNR/SSIM**（训练仍 RGB）[F4]。
- 输出头改为 **从 LR 特征预测局部 k×k 卷积核**（LP-KPN），与残差支路并联 [F3]。

### 明确不做

- UWCL/CX/CoBi 主损失；GAN 主线；在 6k 步上再比 0.03 dB 级 loss 项；无预训练上 20M+。

---

## 5. 对「超分问题本质」的压缩表述

1. **可恢复信息**由 Nyquist 与信噪比决定；其余是先验生成，PSNR 上限由先验与数据匹配度决定 [F1]。  
2. **真实对的主误差源**是 \(T_\tau\) 与 \(\mathcal D(p)\) 空间可变；全局 L1 + 平移不变网把两者都当成「已对齐的固定核」→ 系统性模糊 [F1][F2]。  
3. **在 406 对、8GB、已有 4M 残差网** 的约束下，边际收益排序：  
   **对齐感知训练 ≫ 真实数据/更长 schedule ≫ 局部核头 ≫ 谱/HF/EMA 微调 ≫ 再堆注意力模块。**

---

## 6. Open questions

- Mimicked Alignment 在 **sRGB 已配准** RealSR V3 上的公开绝对数字是否 ≥ SR-RAW 上的幅度（F2 部分为 SR-RAW/ACCV’24）[single source on some rows]。  
- 官方 README V3 ×2 空白；我们应用 **同一 Y 协议** 重报基线后再比。  
- 流估计用冻结 RAFT 级模型 vs 学习 1 层 grid-offset：显存/稳定性待实现后测。

## Sources

见 `findings/F1.md`–`F4.md` 内嵌 URL；关键：RealSR ICCV’19 arXiv:1904.00523；官方 github.com/csjcai/RealSR；Mimicked Alignment ACCV 2024；SDAN/SR-RAW；F4 关于 Y 评测与 KernelGAN。
