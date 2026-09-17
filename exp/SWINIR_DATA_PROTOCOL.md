# SwinIR 用什么数据训练的？—— 以及这为什么重要

**日期：** 2026-09-17
**结论：** 官方 SwinIR 的 real-SR 模型**完全没见过 RealSR**，它训练在 DF2K+OST+WED
（+FFHQ/Manga109/SCUT-CTW1500）上、用 BSRGAN 的高阶合成退化。但**我们的 E11 锚点不是那个模型**，
E11 是我们自己在 RealSR V3 406 对上从零训的 —— 所以目前和 DiT 的对比是公平的。

---

## 1. 官方 SwinIR 的训练数据（来自 github.com/JingyunLiang/SwinIR）

| 任务 | 训练数据 | 退化 |
|------|----------|------|
| classical SR (中) | DIV2K(800) 或 DIV2K+Flickr2K(2650)=DF2K | bicubic |
| lightweight SR | DIV2K | bicubic |
| **real-world SR (SwinIR-M)** | **DIV2K + Flickr2K + OST(10324)** ≈ 1.38 万张 | **BSRGAN (ICCV2021) 高阶退化** |
| **real-world SR (SwinIR-L)** | 上面 + WED(4744) + FFHQ(前2000) + Manga109 + SCUT-CTW1500(前100) ≈ 2 万张 | 同上 |

ckpt 命名直接写明：`003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`
（`D`IV2K `F`lickr2K `O`ST `W`ED `M`anga109 `F`FHQ `S`CUT-`C`TW1500）。

测试集是 `RealSRSet+5images`（无 GT），**不是 RealSR V3 的 100 对**。

## 2. 我们的 E11 锚点

`exp/BASELINE_PROTOCOL.md` 已记录：

> **我们 E11** | RealSR V3 406 对 | ×2 | 真实相机 | Y **33.47** / SSIM 0.9144

E11 = ModSwinIR-base + Align-L1，4.05M，12k 步，**RealSR Train 406 对从零训练**，
外加我们自己加的 OCA/GDFN/AMF/FiLM/DCN 与残差 bicubic 重建。

**所以 E11 与我们的 DiT 是同数据、同倍率、同指标，可以直接横比。** 这点没问题。

## 3. 但这暴露了两个真问题

### (a) 34M 的 DiT 在 406 对上是饿着的

文献里 SwinIR 吃的是 1.4–2 万张图（还是 BSRGAN 合成退化，多样性远高于 406 对真实对）。
我们给 DiT 的只有 406 对。现在 pixel_reg（34M）VAL 33.38 ≈ E11（4.05M）33.47 ——
**8 倍参数换 0 收益**，这很可能不是"DiT 不行"，而是**数据规模撑不起这个容量**。

换句话说：H1/H2 的结论在 **406 对这个 regime 内成立**，但 DiT 的潜力远没释放。

### (b) 若要对标文献的强基线，得走"预训练 + 微调"

标准配方是：DF2K(+OST) 上用 BSRGAN 退化预训练 → RealSR Train 微调。
我们目前 `data/` 下**只有 RealSR(V3)**，没有 DF2K/Flickr2K。

## 4. 建议（按性价比）

1. **短期（不动数据）**：把 E11 当同 regime 锚点，明确写在论文里 ——
   "所有模型均从零训练于 RealSR V3 Train 406 对"，这样对比自洽。
2. **中期（强烈建议）**：下 DF2K（DIV2K 800 + Flickr2K 2650，约 27GB），
   用 BSRGAN 退化合成 LR，**预训练 DiT-XS/S 再在 RealSR 微调**。
   这是最可能把 DiT 拉过 SwinIR 的一步，也顺带补上"大模型需要大数据"的对照。
3. **别做**：拿官方 SwinIR-M/L 的 RealSRSet 数字当我们的 SOTA 对照 —— 数据/退化/测试集全不同。

## 5. 引用时务必核对的协议问题

论文里出现 "SwinIR on RealSR" 时有两种截然不同的东西：

- **(A) RealSR benchmark 协议**：在 RealSR Train 上训练/微调 → 与 E11、与我们的 DiT 可比
- **(B) 零样本**：官方 BSRGAN 预训练模型直接测 RealSR Test → 数据完全不同，通常更差

引用前必须确认是 A 还是 B，否则会拿错锚点。
