# STORY.md · RealSR 真实超分论文汇报

## ① 用户意图对齐

- **目标受众**：CV/ML 学术汇报（同行评审、答辩、组会），观众熟悉 SR、扩散、Transformer，希望看到「实测到底做到什么、为什么"。
- **核心目标**：让观众相信 —— **RealSR ×2 的 PSNR 上限是物理墙（配准），不是模型不够**；我们用「stride-1 U-Net + 正交小波高频损失 + Muon + 移位集成 + TTA」这个配方在 18.7M 下 **34.41 dB / 0.9285 SSIM / 55.96 MUSIQ / 0.3534 MANIQA**，**同协议下全面碾压重训 SwinIR**，**已经坐到 34.4–35.7 dB 的配准受限上限带内**；剩余空间在感知轴（MUSIQ/LPIPS/FID）。
- **PPT 长度**：20 页（封面+目录+1引言+4领域+4数学+1模型+7实验+1结论+1致谢），15–18 分钟报告节奏。
- **视觉调性**：学术蓝白（克制、证据驱动）；强调色（警示红 #D9534F + 数据黄 #FFC107）受配额 ≤ 5%；每页 C 区填充 ≥ 80%；禁大留白/装饰渐变/彩虹。
- **内容边界**：
  - **必讲**：RealSR 配对构造与协议差异；MSE 三堵墙（配准 / 带宽 / 噪声）的实测数据；stride-1 + 小波-HF + 移位 + Muon + TTA 的机制+数据；同协议主表；负结果（Mamba / 扩散 / 双路小波 U-Net / 频域 rectified flow）；感知-失真权衡（Blau & Michaeli）的复现。
  - **不讲**：VAE 底噪细节（只用一句「flux VAE 31.60/0.885」佐证"扩散臂 31.12 是没学到"）；服务器踩坑；git 历史。
  - **禁碰**：跨协议横比（B/C-协议 vs A-strict）；±0.05 dB 单点结论；不出现"我们最 SOTA"等绝对话术。

## ② 页面布局骨架

### 总章节与契约
| 章节 | 页号 | section 扉页 | 编号 |
|---|---|---|---|
| 封面 | 01 | — | — |
| 目录 | 02 | — | — |
| 引言：我们要解决的，是真问题吗？ | 03 | — | — |
| 第一篇：领域状况 | 04–07 | 04（扉页 01） | 01 |
| 第二篇：数学框架 & 我们的思考 | 08–11 | 08（扉页 02） | 02 |
| 第三篇：模型设计 | 12 | — | — |
| 第四篇：实验结果 | 13–19 | 13（扉页 03） | 03 |
| 结论 | 20 | — | — |

### 目录契约
- 目录（02）声明 3 个章节：① 领域状况 ② 数学框架 ③ 实验结果（含模型设计小节）。全篇恰好 3 个 `type: section` 扉页（04 / 08 / 13），编号 01 / 02 / 03 连续，标题与目录逐字一致。

### Hero 页与节奏
- Hero（4 张，占 20%）：01 封面、03 引言巨数、08 章节扉页「三墙」、20 结论收束。
- 至少间隔 1 张 supporting：01→02→03→04，03→04 紧接章节扉页是允许的「transition」。

### rhythm 曲线
| 页 | role | rhythm |
|---|---|---|
| 01 | hero | peak |
| 02 | supporting | valley |
| 03 | hero | peak |
| 04 | transition | transition |
| 05 | supporting | valley |
| 06 | supporting | valley |
| 07 | supporting | peak（领域结论） |
| 08 | hero | peak |
| 09 | supporting | valley |
| 10 | supporting | valley |
| 11 | supporting | peak（数学落点） |
| 12 | supporting | valley（模型最密） |
| 13 | transition | transition |
| 14 | supporting | peak（主表） |
| 15 | supporting | valley |
| 16 | supporting | valley |
| 17 | supporting | valley |
| 18 | supporting | peak（消融） |
| 19 | supporting | peak（权衡复现） |
| 20 | hero | peak |

无连续 ≥3 valley；每章末必有 peak。

### 版式预算
- 非对称版式（必 ≥40%）：03、04、05、07、08、10、11、12、13、14、16、17、18、19、20 → 15/20 = **75%**
- 对称版式（≤2 页）：01 全屏视觉、02 上下分栏目录 → 2/20 ✅
- `N卡片横排` 仅限 02（目录条目），其他页面避免等宽卡片横排。

## ③ 页面大纲（20 页）

---

### 01 · 封面
- **title**：Real-World 单帧图像超分：正交小波高频回归
- **type**：cover
- **role**：hero
- **rhythm**：peak
- **layout**：全屏视觉+大标题
- **visual**：L3 logo + 单位（学术风）
- **visual_role**：atmosphere
- **density**：标题 30字 + 副标 40字 + 4 项 KPI 巨数 + 单位/汇报人/时间
- **anti_pattern**：禁印章/烟花/金装饰/人物大头照；禁大留白撑版面
- **description**：项目名 + 副标题（点题：正交小波高频 + Muon）+ 四项 KPI 大数 + 协议行 · 给评审的「一眼结果」。

---

### 02 · 目录
- **title**：目录
- **type**：catalog
- **role**：supporting
- **rhythm**：valley
- **layout**：上下分栏（左右两列 4+3 条目）
- **visual**：L3 编号方块
- **visual_role**：anchor（编号作为锚点）
- **density**：3 大章 + 7 小节，约 60 字 + 12 个章节编号
- **anti_pattern**：禁圆球编号、繁体数字；禁英文比中文抢眼
- **description**：声明本报告的 3 篇结构，对应章节扉页一一映射。

---

### 03 · 引言：这道题有上界吗？
- **title**：引言：我们测出的，是真上界吗？
- **type**：content
- **role**：hero
- **rhythm**：peak
- **layout**：巨型数字 + 洞察
- **visual**：锚点：34.41 / 0.9285 / 55.96 / 0.3534 四个 KPI 巨字
- **visual_role**：anchor
- **density**：4 巨数 + 1 段 ~80 字引导 + 1 行协议说明
- **anti_pattern**：禁等宽卡片横排；禁把核心数字塞进图表角落；禁 L3 顶替 L1
- **description**：把四指标最大字打到左半；右侧一句话"PSNR 已 34.41，配准受限上界 34.4–35.7 dB → 我们已在内"——报告的钩子。

---

### 04 · 领域篇扉页（01 领域状况）
- **title**：01 · 领域状况：RealSR 是什么、为什么难、谁在做什么
- **type**：section
- **role**：transition
- **rhythm**：transition
- **layout**：巨型数字 + 章节标题 + 4 子要点
- **visual**：大数字 01
- **visual_role**：anchor
- **density**：60 字
- **anti_pattern**：禁满版蓝红金；禁 PART 02 字样泛滥
- **description**：转场。

---

### 05 · RealSR 基准与协议陷阱
- **title**：RealSR：变焦配对基准的"协议陷阱"
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：左大图（数据构造示意）+ 右侧文字 + 下方协议条
- **visual**：左 L1 自绘 SVG：Canon/Nikon 双焦 LR/HR 配对示意
- **visual_role**：evidence
- **density**：约 220 字正文 + 1 张示意 + 1 行协议要点
- **anti_pattern**：禁 50:50 等分双栏；禁把示意缩成 200×70 角标
- **description**：RealSR V3 用 Canon 5D3 / Nikon D810 双焦拍摄并配对；Train 406/390 对、Test 100 对。退化含非平稳模糊+传感器噪声+亚像素错位。**协议敏感性**：A-strict vs B/C 不可横比；本文全部按 A-strict（Limited-range Y、uint8、modcrop4、不 shave、100 对）报数。

---

### 06 · 谱系：判别式 / 退化先验 / 生成式 / 理论
- **title**：领域谱系与定位
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：2×2 关键词矩阵 + 底部定位条
- **visual**：4 张淡描边卡（每张：谱系名 + 代表 + 与我们关系）
- **visual_role**：evidence
- **density**：4 卡 × 60 字 + 底部 1 行定位说明 = 约 260 字
- **anti_pattern**：禁多张深蓝实色卡（多卡页禁用）；禁等宽横排 4 卡
- **description**：判别式（EDSR/RCAN/SwinIR）/ 退化先验（BSRGAN/Real-ESRGAN）/ 生成式（SeeSR/OSEDiff/ResShift/RealSR-R1）/ 理论（Blau&Michaeli 感知-失真权衡 + Baker/Robinson 物理墙）。**我们的定位**：判别式回归 + 同协议全面重训对比。

---

### 07 · 同行对比的诚实边界
- **title**：为什么不能拿"RealSR = 22 dB"的论文数直接比？
- **type**：content
- **role**：supporting
- **rhythm**：peak
- **layout**：上表 + 下结论
- **visual**：表格 4 协议 × N 行（含来源标签）+ 下方 "所以呢" 警示条
- **visual_role**：evidence
- **density**：1 张表（5 列 × 8 行） + 4 行结论 ≈ 280 字
- **anti_pattern**：禁表格无来源；禁跨协议横比无标签
- **description**：A-strict（Limited Y / 不 shave / 100 对，PSNR 偏低）/ B-128（扩散论文 128→512 crop）/ C-REdeg（RealSR 重新合成退化）/ D-blind（未训 RealSR 的官方权重）。**Our 报数严格走 A-strict；与 B/C/D 仅用区间对照、绝不放进同一列**。

---

### 08 · 数学篇扉页（02 数学框架）
- **title**：02 · 数学框架：物理墙、感知权衡、我们要构造的目标
- **type**：section
- **role**：hero
- **rhythm**：peak
- **layout**：巨型数字 + 章节标题 + 4 子要点
- **visual**：大数字 02
- **visual_role**：anchor
- **density**：60 字
- **anti_pattern**：禁满版蓝红金；禁 PART 02 字样泛滥
- **description**：转场到"我们怎么思考"。

---

### 09 · 物理墙 #1：配准 (E_align)—— 真正的天花板
- **title**：MSE 总和 = E_align + E_null + E_noise
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：左公式 + 右实测表 + 下结论
- **visual**：公式 1 行（大） + 4×2 实测表（右） + 底部"34.4–35.7 dB"金句
- **visual_role**：evidence
- **density**：1 公式 + 1 表 + 80 字
- **anti_pattern**：禁公式挤在角落；禁把表当装饰
- **description**：把 MSE 总和拆为三堵物理墙。引用 Baker&Kanade 2002 + Robinson&Milanfar 2004。**实测**：0.2→41.11 / 0.3→37.69 / 0.5→33.41 / 1.0→27.94 dB；真配对残差 0.3–0.5px ⇒ PSNR 物理上限 ≈ **34.4–35.7 dB**。

---

### 10 · 物理墙 #2 / #3：带宽 (E_null) 与传感器噪声 (E_noise)
- **title**：带宽 & 噪声墙（修正"38–40 dB 噪声底"）
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：左表（带宽）×2 + 右表（噪声） + 底部结论
- **visual**：表 1（E_null × 尺度）+ 表 2（E_noise）+ Donoho-MAD 公式（小） + 警示条
- **visual_role**：evidence
- **density**：2 表 + 1 公式 + 90 字
- **anti_pattern**：禁无来源数据；禁把"38–40dB 噪声底"作为正确结论
- **description**：E_null（理想低通）×2=40.65 / ×3=34.55 / ×4=31.27。E_noise σ̂≈0.9/255 ⇒ 49 dB，**比 38–40 dB 谣言高出 9 dB**——噪声不是 binding 墙，**配准是**。引用 Chatterjee & Milanfar 2010 + Donoho MAD。

---

### 11 · 数学落点：MSE 最优解是条件均值 ⇒ 高频被抹平
- **title**：L1/L2 最优解 = 条件均值 ⇒ 高频被抹平 → 需要正交局部机制
- **type**：content
- **role**：supporting
- **rhythm**：peak
- **layout**：左公式链（大） + 右洞察 + 底部金句条
- **visual**：3 行公式（arg min E‖ŷ−y‖₁ = E[y|x]） + Parseval 等式（小） + 核心金句条
- **visual_role**：anchor
- **density**：3 公式 + 130 字
- **anti_pattern**：禁公式挤边；禁金句条放底但不锚
- **description**：L1/L2 的 min-MSE 解 = 条件均值 E[y|x]，对自然图像即低通。**Parseval ⇒ 小波高频子带 L1 是像素 L1 的无损重分配**，不会把梯度从 LL 偷走。**故在 L1 + 正交小波高频子带 L1(λ) 上抬高频，零推理成本**。这一步把数学动机落实成具体的损失公式。

---

### 12 · 模型设计：配方一张图
- **title**：完整配方：stride-1 U-Net + λ8 小波 HF + 移位集成 + Muon + BSRGAN + TTA
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：上中下三段：骨干（左） + 损失（中公式 + 子卡） + 推理（TTA）
- **visual**：左 U-Net 简笔 SVG + 中损失公式 + 右 D4-TTA 示意图（小）+ 下方 6 块小卡（Muon / 预训练 / 零初始化头 / EMA / 10000 步 / batch128）
- **visual_role**：evidence（模型图）+ anchor（损失公式）
- **density**：1 骨干示意 + 1 损失公式 + 1 TTA 示意 + 6 子卡 ≈ 320 字
- **anti_pattern**：禁把 6 子卡堆成等宽横排；禁公式挤一行
- **description**：骨干（stride-1 + base 64 + mult 1,2,4,4 + 2 res + attn@Lv2-3 + 零初始化头 / step0=bicubic）。损失（L1 + 多尺度 DWT HL/LH/HH λ8 + 移位集成 {0,1,2,3} dyadic offsets）。Muon（≥2D NS5 正交化，1-D AdamW，lr 5e-3）。BSRGAN 预训练 init。EMA 0.999。D4-TTA 8 重。**这张图是配方的 single source of truth**。

---

### 13 · 实验篇扉页（03 实验结果）
- **title**：03 · 实验结果：主表、上限、消融、负结果、权衡
- **type**：section
- **role**：transition
- **rhythm**：transition
- **layout**：巨型数字 + 章节标题 + 5 子要点
- **visual**：大数字 03
- **visual_role**：anchor
- **density**：60 字
- **anti_pattern**：禁满版蓝红金；禁 PART 02 字样泛滥
- **description**：转场到"我们做到了什么"。

---

### 14 · 主表：同协议 × 全指标 × 含 Ours
- **title**：主表 · RealSR ×2 · A-strict 协议 · 100 对均值
- **type**：content
- **role**：supporting
- **rhythm**：peak
- **layout**：上全表 + 下 4 指标金句条
- **visual**：表格 7 列 × 11 行（A 组含 Ours / 锚点 / 重训基线 / 地板 / 我们生成式自臂）+ 底部"四指标全最优 + +1.44 dB / +9.2 MUSIQ / +0.048 MANIQA vs SwinIR"金句
- **visual_role**：anchor
- **density**：1 大表 + 1 金句 ≈ 220 字
- **anti_pattern**：禁跨协议混行；禁无 Ours 行；禁列名漏 PSNR/SSIM/MUSIQ/MANIQA/Params
- **description**：A 组全行：Ours(s1_b64+TTA) 34.41/.9285/55.96/.3534；s1_b64 anchor 34.11/.9246/55.22/.342；EDSR 33.60；E11 33.47；SwinIR-largeish 32.97；SwinIR-light 32.88；Bicubic 31.73；**我们 latent_flow 自臂 28.36/.790/50.12/.224（保真感知换位的同协议对照点）**。底部 B/C 组用区间表，但**不放进主表**。

---

### 15 · 物理上界：实测三墙 + 我们在内
- **title**：免训练实测 · 三堵物理墙 ⇒ ×2 PSNR 上界 34.4–35.7 dB
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：左大图（fig_ceiling.png）+ 右表 + 底部金句条
- **visual**：L1 = docs/figs/fig_ceiling.png（已在 assets/） + 右表（三墙 × 尺度）
- **visual_role**：anchor（图）+ evidence（表）
- **density**：1 大图 + 1 表 + 100 字
- **anti_pattern**：禁大图撑满 720 但不留金句；禁把图当装饰
- **description**：左图：注册误差-PSNR 曲线 + 三尺度墙柱 + 我们的红菱形 **34.41 在带内**。右表：×2=40.1（无配准）→ 35.7（+0.3px）→ 34.4（+0.4px）；×3 / ×4 同步展示。**「不是模型不够，是物理不允许」——这正是论文的卖点**。

---

### 16 · 跨尺度：×2 已顶，×3 欠调，×4 近顶
- **title**：×2 / ×3 / ×4 的上限 + 我们实测 + scale-adaptive λ
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：上表（3 尺度 × 2 列：上限 vs Ours）+ 下表（scale-adaptive λ 扫描）+ 底部洞察
- **visual**：表 1（3×2）+ 表 2（3×4 λ 扫描）+ 洞察条
- **visual_role**：evidence
- **density**：2 表 + 90 字
- **anti_pattern**：禁把"×3 欠调"藏起来；禁只展示成功尺度
- **description**：×2 34.41 vs 上限 35.7 ⇒ **见顶**；×3 31.13 vs 34.4 ⇒ **欠调仍有 3+ dB 真实空间**；×4 29.51 vs 31.2 ⇒ **近顶**。**scale-adaptive λ**：×2≈8 / ×3≈2 / ×4≈5；×3 重退化 ⇒ HF 更难恢复 ⇒ λ 须收。

---

### 17 · 负结果 · 1：扩散 / Mamba / 双路小波 / 频域流
- **title**：负结果 · 一（扩散 / Mamba / 双路小波 / 频域 rectified flow）
- **type**：content
- **role**：supporting
- **rhythm**：valley
- **layout**：上表（5 路线 × 结论 + 关键数据）+ 下结论
- **visual**：表（5 行）+ 底部「统一洞见：金句条」
- **visual_role**：evidence
- **density**：1 表 + 90 字
- **anti_pattern**：禁只展示赢的；禁把负结果藏在脚注
- **description**：**扩散**（pixel/latent × flow/reg）：所有 flow < bicubic 31.73；NFE64 饱和到 ~29.1；latent_reg 完全没学起来（< 自身零初始化）。**Mamba/VSS**（等预算 328k 样本）：Mamba 31.68 vs U-Net 33.97；Mamba 给 4.7× 预算才 33.42 仍 < U-Net。**双路小波 U-Net**（z_dual*）：33.85 远低于 b64+λ8 的 34.24；架构性"分频"敌不过 loss-only。**频域 rectified flow**（refine 头）：scale 0.10→31.07 / 0.20→27.18 / 0.35→23.19，单调毁图。**统一洞见**：在 390 对规模下判别式回归吃满可预测部分；"生成不可预测高频"的路线均告负。

---

### 18 · 消融 · 同协议 ΔPSNR 一图看懂
- **title**：消融 · 同协议单因子 ΔPSNR
- **type**：content
- **role**：supporting
- **rhythm**：peak
- **layout**：左大图（fig_ablation.png）+ 右上下两表（×2 收益排序 + 跨尺度 λ）+ 底部洞察条
- **visual**：L1 = docs/figs/fig_ablation.png + 右表（收益排序）
- **visual_role**：anchor
- **density**：1 大图 + 2 表 + 120 字
- **anti_pattern**：禁消融只列数字不排序；禁把收益归并成「+0.x dB 综合」
- **description**：stride-2→1 **+0.38**（最大单项） > Muon 5e-3 **+0.15** ≈ 小波 λ8 **+0.11** > TTA **+0.10** ≈ BSRGAN 预训练 **+0.09** > 容量 5M→72M **+0.05**（饱和）。**架构 ≫ 机制loss > 优化器 > 数据 > 步数 > 容量**。**×3/×4 用 scale-adaptive λ（2/5）**——单一 λ 无法通用。

---

### 19 · 感知-失真权衡：Blau & Michaeli 的同协议复现
- **title**：感知-失真权衡 · Blau & Michaeli 2018 · 同协议复现
- **type**：content
- **role**：supporting
- **rhythm**：peak
- **layout**：左大图（fig_tradeoff.png）+ 右两卡（回归 / 生成式）+ 底部金句条
- **visual**：L1 = docs/figs/fig_tradeoff.png + 右两卡（"高保真" / "高感知"）+ 金句条
- **visual_role**：anchor
- **density**：1 大图 + 2 卡 + 100 字
- **anti_pattern**：禁把生成式当成我们主线；禁把图当装饰
- **description**：**我们的回归 34.41 / 0.9285 / 55.96 / 0.3534** 与 **我们的生成式 latent_flow 28.36 / 0.790 / 50.12 / 0.224**：注入高频换 6 dB PSNR；MUSIQ/MANIQA 仍分裂（生成式 MUSIQ 高 MANIQA 低）。**这正是 Blau & Michaeli 在 RealSR 的具体化**：fidelity 边界由物理墙定，realism 边界由数据量定。**下一步只剩感知轴**。

---

### 20 · 结论
- **title**：结论 · 配方、上界、与下一步
- **type**：content
- **role**：hero
- **rhythm**：peak
- **layout**：上三栏卡（"配方" / "上界" / "下一步"）+ 下复现命令
- **visual**：3 卡（深蓝实色单页限 1，张数=3 时降为淡描边）+ 底部 reproduce 代码块
- **visual_role**：anchor（3 卡数字）
- **density**：3 卡 × 80 字 + 1 复现命令 = 280 字
- **anti_pattern**：禁满版红金；禁"我们 SOTA"绝对话术；禁复现命令用截图
- **description**：**配方**：stride-1 + λ8 + 移位 + Muon + 预训练 + TTA → **34.41 / 0.9285 / 55.96 / 0.3534**，同协议四指标全面超重训 SwinIR (+1.44 dB / +9.2 MUSIQ / +0.048 MANIQA)，18.7M。**上界**：×2 已落 34.4–35.7 dB 配准受限带内；剩余空间在感知轴。**下一步**：×3/×4 全配方逼近上限；保真见顶 → 转 MUSIQ/LPIPS/FID。复现命令：python -m diffusion.train_pixel ... + python -m diffusion.oracle --scale 2。

---

## 数据必须落点（信息板 → 汇报稿）

| 页 | 关键数字 | 落点判断 |
|---|---|---|
| 03 | 34.41 dB / 0.9285 SSIM | "配准受限上界 34.4–35.7 dB → 已在内" |
| 07 | 协议 4 类差 ~0.4–1 dB | "论文数不可平移；本文只横比 A-strict" |
| 09 | 0.3 px → 37.69 / 0.5 px → 33.41 | "配准是真墙，0.3–0.5 px 残差 ⇒ 34.4–35.7 上界" |
| 10 | σ̂ ≈ 0.9 ⇒ 49 dB | "修正 38–40 dB 噪声底谣言：噪声不是墙" |
| 11 | argmin E‖ŷ−y‖₁ = E[y\|x] | "L1 最优解是条件均值 ⇒ 高频被抹 ⇒ 需要正交局部机制" |
| 12 | stride-1 / λ8 / Muon 5e-3 | "配方 single source of truth：6 子项彼此独立可累加" |
| 14 | +1.44 dB vs SwinIR | "同协议四指标全最优，参数量更小" |
| 15 | 34.41 落在带内 | "不是模型不够，是物理不允许" |
| 16 | ×3 31.13 vs 34.4 | "×3 欠调仍有 3+ dB 真实空间（区别于 ×2 见顶）" |
| 17 | flow < bicubic / Mamba -2.3 dB | "在 390 对规模下判别式吃满，生成不可预测高频路线均告负" |
| 18 | 收益排序：stride-1 > Muon > λ8 | "架构 ≫ 机制 > 优化器 > 数据 > 容量" |
| 19 | 28.36 / 0.790 生成式 vs 34.41 / 0.9285 | "Blau & Michaeli 在 RealSR 的具体化：fidelity 由物理墙定，realism 由数据量定" |
| 20 | 全项 34.41 / 0.9285 / 55.96 / 0.3534 | "保真见顶 → 转感知轴；×3/×4 是真实下一步" |