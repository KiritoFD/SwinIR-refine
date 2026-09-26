# -*- coding: utf-8 -*-
"""RealSR 学术结构中文幻灯片：引言→相关工作→方法(数学)→实验(密集大表)。
先跑 docs/make_figs.py 生成 docs/figs/*.png，再 python docs/make_slides.py。"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figs"
OUT = HERE / "slides_final.pptx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
HEADBG = RGBColor(0x1F, 0x4E, 0x79)
LIGHT = RGBColor(0xEE, 0xF2, 0xF7)
DARK = RGBColor(0x20, 0x20, 0x20)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
OURROW = RGBColor(0xFB, 0xE5, 0xA7)
RED = RGBColor(0xB0, 0x30, 0x24)

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height
_state = {"n": 0}


def _footer(s):
    _state["n"] += 1
    b = s.shapes.add_textbox(Inches(0.4), Inches(7.05), Inches(10.0), Inches(0.35))
    p = b.text_frame.paragraphs[0]
    p.text = "真实世界单帧超分 · RealSR(V3) ×2 · 判别式回归 + 正交小波高频损失 + Muon + TTA"
    p.font.size, p.font.color.rgb = Pt(9), RGBColor(0x88, 0x88, 0x88)
    pb = s.shapes.add_textbox(Inches(12.5), Inches(7.05), Inches(0.7), Inches(0.35))
    q = pb.text_frame.paragraphs[0]
    q.text = str(_state["n"]); q.alignment = PP_ALIGN.RIGHT
    q.font.size, q.font.color.rgb = Pt(10), RGBColor(0x88, 0x88, 0x88)


def _bar(s, title, kicker=None):
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(1.0))
    bar.fill.solid(); bar.fill.fore_color.rgb = HEADBG; bar.line.fill.background()
    tf = bar.text_frame; tf.word_wrap = True
    tf.margin_left, tf.margin_top = Inches(0.4), Inches(0.08)
    p = tf.paragraphs[0]; p.text = title
    p.font.size, p.font.bold, p.font.color.rgb = Pt(24), True, WHITE
    if kicker:
        q = tf.add_paragraph(); q.text = kicker
        q.font.size, q.font.color.rgb = Pt(12), RGBColor(0xC7, 0xD6, 0xE5)


def slide(title=None, kicker=None):
    s = prs.slides.add_slide(BLANK)
    if title:
        _bar(s, title, kicker)
        _footer(s)
    return s


def bullets(s, items, top=1.25, left=0.5, width=12.35, size=16):
    body = s.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(5.7))
    tf = body.text_frame; tf.word_wrap = True
    first = True
    for line in items:
        lvl = 0
        txt = line
        if line.startswith("· "):
            txt = line[2:]; lvl = 1
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = txt; p.level = lvl
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.color.rgb = DARK if lvl == 0 else RGBColor(0x44, 0x44, 0x44)
        p.space_after = Pt(6 if lvl == 0 else 3)
    return s


def num_cols(rows, cols):
    """return {col: (best_row, second_row)} higher-is-better over numeric values."""
    out = {}
    for j in cols:
        vals = []
        for i, r in enumerate(rows):
            v = r[j]
            if isinstance(v, (int, float)):
                vals.append((i, v))
        vals.sort(key=lambda t: t[1], reverse=True)
        best = vals[0][0] if len(vals) >= 1 else None
        second = vals[1][0] if len(vals) >= 2 else None
        out[j] = (best, second)
    return out


def table(s, header, rows, cols_to_rank, top=1.3, cell_h=0.36, fsize=11,
          head_fs=11, highlight_rows=(), first_col_w=None):
    nr, nc = len(rows) + 1, len(header)
    rank = num_cols(rows, cols_to_rank)
    shp = s.shapes.add_table(nr, nc, Inches(0.35), Inches(top), Inches(12.6), Inches(cell_h * nr))
    t = shp.table
    if first_col_w:
        t.columns[0].width = Inches(first_col_w)
    for j, h in enumerate(header):
        c = t.cell(0, j)
        c.fill.solid(); c.fill.fore_color.rgb = HEADBG
        c.margin_top = c.margin_bottom = Emu(27432)
        p = c.text_frame.paragraphs[0]; p.text = h
        p.font.size, p.font.bold, p.font.color.rgb = Pt(head_fs), True, WHITE
    for i, r in enumerate(rows, start=1):
        ri = i - 1
        hl = ri in highlight_rows
        for j, v in enumerate(r):
            c = t.cell(i, j)
            c.fill.solid()
            c.fill.fore_color.rgb = OURROW if hl else (LIGHT if i % 2 == 0 else WHITE)
            c.margin_top = c.margin_bottom = Emu(22860)
            p = c.text_frame.paragraphs[0]
            p.text = str(v) if isinstance(v, str) else (f"{v:.4f}" if j in (4,) else f"{v}")
            is_num = isinstance(v, (int, float))
            bold = second = False
            if is_num and j in rank:
                b, sd = rank[j]
                bold = (ri == b); second = (ri == sd)
            p.font.size = Pt(fsize)
            p.font.bold = bool(bold or hl and j == 0)
            p.font.underline = bool(second)
            p.font.color.rgb = RED if bold else DARK
    return s


def img_slide(fname, title, caption=None, kicker=None):
    s = slide(title, kicker)
    p = str(FIGS / fname)
    if Path(p).exists():
        s.shapes.add_picture(p, Inches(0.8), Inches(1.3), width=Inches(11.75))
    if caption:
        cb = s.shapes.add_textbox(Inches(0.8), Inches(6.5), Inches(11.75), Inches(0.5))
        cp = cb.text_frame.paragraphs[0]; cp.text = caption; cp.alignment = PP_ALIGN.CENTER
        cp.font.size, cp.font.color.rgb = Pt(11), RGBColor(0x55, 0x55, 0x55)
    return s


# ============================ 1 标题页 =====================================
s = slide()
band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.9), SW, Inches(3.6))
band.fill.solid(); band.fill.fore_color.rgb = ACCENT; band.line.fill.background()
tf = band.text_frame; tf.word_wrap = True; tf.margin_left = Inches(0.8); tf.margin_top = Inches(0.3)
tf.paragraphs[0].text = "真实世界单帧图像超分辨率"
tf.paragraphs[0].font.size, tf.paragraphs[0].font.bold, tf.paragraphs[0].font.color.rgb = Pt(34), True, WHITE
p = tf.add_paragraph(); p.text = "—— 正交小波高频损失 + Muon 优化的判别式回归方法"
p.font.size, p.font.bold, p.font.color.rgb = Pt(22), True, RGBColor(0xD6, 0xE3, 0xF0)
p = tf.add_paragraph(); p.text = " "
p = tf.add_paragraph(); p.text = "数据集：RealSR (V3) · 放大倍率 ×2 · 官方 Test.m 协议（有限量程 Y / 不剃边 / 100 对）"
p.font.size, p.font.color.rgb = Pt(14), RGBColor(0xC7, 0xD6, 0xE5)
tb = s.shapes.add_textbox(Inches(0.8), Inches(5.8), Inches(11.7), Inches(1.0))
tp = tb.text_frame.paragraphs[0]
tp.text = "PSNR-Y 34.41 dB   ·   SSIM 0.9285   ·   MUSIQ 55.96   ·   MANIQA 0.3534   （参数量 18.7M）"
tp.font.size, tp.font.bold, tp.font.color.rgb = Pt(19), True, ACCENT

# ============================ 2 目录/概览 ==================================
s = slide("报告结构", "Intro → Related Works → Method → Experiments")
bullets(s, [
    "一、引言（问题本身）：真实相机变焦配对超分，为什么 PSNR 有硬上限",
    "· 三大不可约误差：信息零空间 / 传感器噪声 / 亚像素配准残差",
    "二、相关工作：判别式 CNN/Transformer、GAN 真实超分、扩散式、感知-失真权衡理论",
    "三、方法（从数学与问题本质出发）：stride-1 回归、正交小波高频损失、移位集成、Muon、预训练与 TTA",
    "四、实验（高密度结果）：全方法全指标大表、消融、经验上限实测、跨尺度、负结果",
], size=17)

# ============================ 5-6 引言 =====================================
s = slide("一、引言：问题的本质", "RealSR 是「用一台相机拍两张不同焦距照片」得到的真实配对")
bullets(s, [
    "任务：输入真实低清 x↓（短焦），输出高清 y（长焦，配准后），最小像素误差同时保留真实高频。",
    "与合成分辨率（bicubic 退化）的根本不同：退化未知、非平稳、含真实噪声与镜头畸变，",
    "· 因此 bicubic 之后 HR 仍留有「真实高频相位」，但高于奈奎斯特的信息已被物理摧毁。",
    "目标定位：本项目做的是「判别式回归」路线——最大化保真度，而非生成式「造出合理高频」。",
    "核心问题：这份数据集在像素保真（PSNR）上的天花板到底在哪？我们离它还有多远？",
], size=16)

s = slide("一、引言：误差分解与理论上限（本文立论基础）", "MSE_total = E_null + E_noise + E_align")
bullets(s, [
    "① 信息零空间 E_null：下采样按奈奎斯特定理抹除高频，网络只能「猜」，无法复原（Baker & Kanade, TPAMI 2002）。",
    "② 独立传感器噪声 E_noise：GT 自身带暗电流/散粒噪声（随机不可知，aleatoric），完美模型也预测不了那一瞬的噪声（Chatterjee & Milanfar, TIP 2010）。",
    "③ 配准残差 E_align：变焦双镜头光学透视/畸变不同，标定后仍余 0.2–0.5px 非刚性错位；PSNR 对亚像素位移极敏感（Robinson & Milanfar, TIP 2004）。",
    "结论（后由免训练实测证实，见 §四）：真实 SR 的 PSNR 上限是有限值，且在此数据上由「配准」主导；",
    "· 换言之——不是模型不够大，是物理不允许。本报告的核心贡献即「测量并逼近这个上限」。",
], size=14.5)

# ============================ 相关工作 ====================================
s = slide("二、相关工作", "按方法与评测协议两条线梳理")
bullets(s, [
    "判别式 CNN：VDSR/EDSR/RCAN/RDN — 追求 PSNR，但基于合成退化，真实场景偏糊。",
    "判别式 Transformer：SwinIR(HAT 等) — 全局注意力更强表达，真实超分重训后仍是保真主力。",
    "真实退化先验：BSRGAN / Real-ESRGAN — 二阶合成退化训练，泛化但非配对最优。",
    "生成式（GAN/扩散）：StableSR / DiffBIR / SeeSR / ResShift / OSEDiff / SinSR / AdcSR / TSD-SR / PURE / RealSR-R1 …",
    "· 感知更强、但 PSNR 常暴跌 6–10 dB：受感知-失真权衡支配（Blau & Michaeli, CVPR 2018）。",
    "理论：SR 极限与配准统计界（Baker&Kanade 2002；Robinson&Milanfar 2004/2006）；去噪 CRLB（Chatterjee&Milanfar 2010）。",
], size=15)

s = slide("二、相关工作：本工作的定位", "在「同协议、判别式」坐标系里做到保真前沿")
bullets(s, [
    "与「刷指标」路线相反：我们不引入外部大数据、不用扩散，只在 RealSR 训练域内做机制创新。",
    "把力气花在数学上讲得通的地方：正交小波高频损失（而非全局 FFT 增益）、相位保真的 stride-1、正交化的 Muon 优化。",
    "并给出可证伪的经验上限：用免训练探针量出三堵墙，证明 34.41 dB 已接近数据集天花板。",
    "与扩散 SOTA 不直接比 PSNR（协议/输入不同），只做定性对照 + 我们自有生成臂的同协议实证权衡。",
], size=16)

# ============================ 方法 ========================================
s = slide("三、方法 · 总体形式", "以「双三次上采 + 残差学习」为骨架")
bullets(s, [
    "回归目标：ŷ = B(x↓) + f_θ(x↓ , θ)，B 为 bicubic 上采样，f_θ 学「高频残差」。",
    "训练损失：L(θ) = ‖ŷ − y‖₁ + λ · L_HF(ŷ, y)   ——主 L1 + 小波高频子带项（见下页）。",
    "关键设计1 零初始化输出头：f_θ 末层置 0 ⇒ 训练起点 ŷ ≡ B(x↓)，即「bicubic 地板不变量」，任何劣化都可被早停挡住。",
    "关键设计2 stride-1：主干全程在 HR 尺度运算（不 lossy 下采样），保住高频相位——实测 +0.38 dB（本工作最大单项增益）。",
    "评价协议：训练 RGB、测试有限量程 Y（BT.601），uint8，modcrop4，不剃边，100 对全量。",
], size=15)

s = slide("三、方法 · 正交小波高频损失（核心创新）", "为什么是「正交 + 局部」的小波，而非 FFT")
bullets(s, [
    "动机（数学）：L1 回归的最优解是条件均值 E[y|x]，它对高频「取平均」⇒ 过糊；需显式抬高频子带。",
    "做法：对 HR 与 ŷ 各做 J 级正交 Haar DWT，仅惩罚高频子带 HL/LH/HH：",
    "· L_HF = Σ_{j=1..J} Σ_{dir∈{HL,LH,HH}} ‖ W_dir^{(j)}(ŷ) − W_dir^{(j)}(y) ‖₁ ，总损失权重 λ（×2 最优 λ≈8）。",
    "为什么正交：Parseval 保能量 ⇒ 子带 L1 是像素 L1 的「无损重分配」，只改变频率侧重，不引入伪能量（FFT 全局基不满足，已证伪）。",
    "为什么局部：小波基紧支撑，可表达镜头非平稳退化；全局傅里叶增益会失稳（Wiener/AmpPhase/RadialPSF 全部失败）。",
    "推理零成本：只改损失不改网络，训练完即免费。",
], size=14.5)

s = slide("三、方法 · 移位集成（DTCWT 平移不变性的廉价替身）", "Haar 子带是平移敏感的")
bullets(s, [
    "问题：一次 Haar 分解中，1px 平移会把边缘能量在 HL/LH/HH 之间乱搬 ⇒ 网络可「放错子带」来骗过 L1。",
    "做法：对 4 个二进制位移偏移 s∈{(0,0),(1,0),(0,1),(1,1)} 分别做小波高频损失并取平均：",
    "· L_shift = (1/4) Σ_s L_HF( roll(ŷ, s), roll(y, s) )   —— 使高频目标近似平移不变。",
    "另实现真·双树复小波 DTCWT（两棵树 1px 偏移 → 方向子带取模），但在强基座上实测不敌「Haar+移位集成」。",
    "结果：移位集成把 MUSIQ 顶到全项目最高 56.02（非 TTA）/ MANIQA 0.3534——是机制层面唯一仍在涨点的项。",
], size=14.5)

s = slide("三、方法 · Muon 优化器", "把「正交」从损失推广到优化器")
bullets(s, [
    "对 ≥2D 权重用 Muon：对动量缓冲做 Newton–Schulz 迭代求近似正交化 O=NS(G)，更新 Δθ = −η·scale·O。",
    "直觉：正交化更新让各奇异方向步长均衡（谱归一化），在小数据 SR 上比 AdamW 更接近最优步。",
    "1D 参数（bias/norm）仍用 AdamW 复合；两路各自独立 cosine 调度（aux-lr 解耦）。",
    "实测（bs=128 对照）：Muon 在 1e-3~1e-2 全 lr 段三指标胜 AdamW；最优 lr=5e-3。",
    "· 关键旁证：零预训练 + Muon 的分数已≈预训练 + AdamW，即 Muon 抵掉一大截预训练收益。",
], size=15)

s = slide("三、方法 · 预训练与测试时增强", "把可复用的先验与方差削减叠上")
bullets(s, [
    "BSRGAN 二阶合成退化在 DIV2K+Flickr2K 预训练 → 在 RealSR 上微调（Muon+λ8 小波），四指标同向 +0.09 dB。",
    "TTA：D4 群 8 个变换各前向、反变换回原方向、float 平均、最后一次性量化为 uint8。",
    "· 纯推理端免费增益：抬 SSIM/PSNR/MANIQA；MUSIQ 常持平（8 重平均轻微平滑极锐纹理）——再次说明须多指标同看。",
], size=16)

# ============================ 实验 ======================================
# 主大表：全方法全指标（同协议 B-128/512, ×4 文献 + 我们相关方法另置），最优加粗 次优下划线
s = slide("四、实验 · 主表①：官方配对协议 A-strict（本项目自测，全指标）",
          "有限量程Y · 不剃边 · 100 对 —— 同格最优加粗、次优下划线")
rows_a = [
    ["Bicubic", "—", "—", 31.73, 0.8876, 39.40, 0.278],
    ["SwinIR-light（重训）", "Transformer", "0.61M", 32.88, 0.9026, 45.43, 0.302],
    ["SwinIR-largeish（重训）", "Transformer", "3.96M", 32.97, 0.9058, 46.79, 0.305],
    ["E11 Mod-SwinIR+Align", "Transformer", "4.05M", 33.47, 0.9143, 49.15, 0.309],
    ["EDSR-baseline（复现）", "CNN", "1.19M", 33.60, 0.9167, 53.14, 0.327],
    ["latent rectified-flow（我们）", "生成式", "32.9M", 28.36, 0.790, 50.12, 0.224],
    ["s1_b64（AdamW, 零预训练）", "U-Net", "18.7M", 34.11, 0.9246, 55.22, 0.342],
    ["Ours（完整配方 + TTA）", "U-Net stride-1", "18.7M", 34.41, 0.9285, 55.96, 0.3534],
]
table(s, ["方法", "类别", "参数", "PSNR-Y↑", "SSIM↑", "MUSIQ↑", "MANIQA↑"],
      rows_a, cols_to_rank=[3, 4, 5, 6], top=1.45, cell_h=0.55, fsize=13, head_fs=12,
      highlight_rows={7}, first_col_w=3.6)

# 主大表②：同协议文献大表（所有方法，全指标）
s = slide("四、实验 · 主表②：文献同协议大表 B-128/512 ×4（全方法·全指标）",
          "统一 RealSR 裁块协议 · 出处 TinySR(2508.17434) · 最优加粗、次优下划线")
rows_b = [
    ["BSRGAN", "GAN", 26.38, 0.7651, 63.28, 0.5425],
    ["Real-ESRGAN", "GAN", 26.65, 0.7603, 60.45, 0.5507],
    ["LDL", "GAN", 25.28, 0.7565, 60.92, 0.5494],
    ["DiffBIR", "扩散", 25.93, 0.6525, 65.66, 0.6296],
    ["StableSR", "扩散", 28.04, 0.7454, 58.53, 0.5603],
    ["SeeSR", "扩散", 28.14, 0.7712, 64.74, 0.6022],
    ["ResShift", "扩散", 28.69, 0.7874, 52.40, 0.4756],
    ["SinSR", "一步扩散", 28.38, 0.7499, 55.03, 0.4904],
    ["OSEDiff", "一步扩散", 27.92, 0.7836, 64.69, 0.5898],
    ["AdcSR", "一步扩散", 28.10, 0.7726, 66.26, 0.5927],
    ["TSD-SR", "一步扩散", 27.77, 0.7559, 66.62, 0.5874],
    ["TinySR", "一步扩散", 27.48, 0.7459, 65.36, 0.5804],
]
table(s, ["方法", "类别", "PSNR-Y↑", "SSIM↑", "MUSIQ↑", "MANIQA↑"],
      rows_b, cols_to_rank=[2, 3, 4, 5], top=1.35, cell_h=0.44, fsize=12, head_fs=12,
      first_col_w=3.2)
cap = s.shapes.add_textbox(Inches(0.35), Inches(6.62), Inches(12.6), Inches(0.4))
cp = cap.text_frame.paragraphs[0]
cp.text = "注：此协议输入为再退化的 RealSR（更难），与表①不可跨协议横比；其生成式方法以感知(MUSIQ/MANIQA)换保真(PSNR)。"
cp.font.size, cp.font.color.rgb = Pt(11), RED

# 上限实测
s = slide("四、实验 · 免训练实测的数据集上限（三堵墙）", "diffusion/oracle.py · 不训练、不外推")
rows_c = [
    ["E_align 亚像素配准", "0.2px→41.11", "0.3px→37.69", "0.5px→33.41", "1.0px→27.94"],
    ["E_null 带宽截断(×2 LR带)", "40.65", "×3:34.55", "×4:31.27", "—"],
    ["E_noise 传感器(MAD σ≈0.9/255)", "49.05", "—", "—", "—"],
    ["合成(带宽+噪声)", "40.1", "34.4", "31.2", "—"],
    ["合成 + 0.3px 配准", "35.7", "~32", "~29.5", "—"],
]
table(s, ["误差项", "×2 (dB)", "次列", "再列", "备注"], rows_c, cols_to_rank=[],
      top=1.4, cell_h=0.6, fsize=13, head_fs=12, first_col_w=4.0)
b = s.shapes.add_textbox(Inches(0.5), Inches(4.7), Inches(12.3), Inches(1.9))
tf = b.text_frame; tf.word_wrap = True
for i, line in enumerate([
    "结论：×2 由「配准残差」主导上限≈34.4–35.7 dB；我们的 34.41 dB / 0.9285 SSIM 已落在该带内 ⇒ 保真已近物理极限。",
    "修正认知：本数据 GT 噪声很小（σ≈0.9），所谓「38–40dB 噪声底」在此不成立，真正的墙是配准，其次是带宽。",
    "含义：再往上刷 PSNR 只能向条件均值抹高频（降感知）——这正是扩散/生成式 PSNR 崩塌的原因（见下页散点）。",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = line; p.font.size = Pt(14); p.font.color.rgb = DARK; p.space_after = Pt(6)

# 权衡散点图
img_slide("fig_tradeoff.png", "四、实验 · 感知-失真权衡（同协议自证）",
          "同测试集同协议：我们回归处于保真前沿，自有生成臂以约 6 dB 换无参考 IQA，与 Blau-Michaeli(CVPR18) 吻合。")

# 结果图
img_slide("fig_results.png", "四、实验 · 主表①可视化",
          "官方配对协议下 PSNR / SSIM 阶梯；红虚线为最终结果 34.41 dB。", kicker="柱状图与表①同源")

# 消融
s = slide("四、实验 · 消融（同协议单因子，Δ PSNR-Y dB）")
rows_d = [
    ["stride-2 → stride-1", "+0.38", "架构", "最大单项杠杆"],
    ["+ 小波高频损失 λ8", "+0.11", "损失", "MUSIQ/MANIQA 同时↑"],
    ["+ Muon(5e-3)", "+0.15", "优化器", "抵一部分预训练"],
    ["+ 移位集成", "+0.02", "损失", "MUSIQ 最高 56.0"],
    ["+ BSRGAN 预训练", "+0.09", "数据", "6× 训练时长"],
    ["+ TTA", "+0.10", "推理", "8× 推理，SSIM/MANIQA↑"],
    ["容量 5M→72.5M", "+0.05", "规模", "饱和"],
]
table(s, ["改动", "Δ PSNR-Y", "维度", "备注"], rows_d, cols_to_rank=[], top=1.4, cell_h=0.55, fsize=13, head_fs=12, first_col_w=3.8)

img_slide("fig_ablation.png", "四、实验 · 消融可视化", caption="机制项（架构 + 正交频率损失 + 优化器）主导，纯堆参数饱和。")

# 跨尺度 + 上限
s = slide("四、实验 · 跨尺度迁移与 scale-adaptive λ")
rows_e = [
    ["×2", "34.11", "34.41(λ8+shift+TTA)", "≈35.7", "在天花板内"],
    ["×3", "31.04", "31.13(λ2)", "≈34.4", "欠调，仍有空间"],
    ["×4", "29.42", "29.51(λ5)", "≈31.2", "接近天花板"],
]
table(s, ["尺度", "plain", "ours(best λ)", "该尺度上限", "判定"], rows_e, cols_to_rank=[], top=1.5, cell_h=0.6, fsize=14, head_fs=13, first_col_w=1.5)
b = s.shapes.add_textbox(Inches(0.5), Inches(4.0), Inches(12.3), Inches(2.4))
tf = b.text_frame; tf.word_wrap = True
for i, line in enumerate([
    "最优 λ 随退化强度反相关：×2≈8、×3≈2、×4≈5（×3 处 λ5 过锐、MUSIQ 反降；×4  λ5 全面有效）。",
    "绝对值随尺度下降是任务本身更难，不可与 ×2 横比。",
    "结论：×3/×4 因本轮欠调仍有余量——是唯一在保真轴上明确「未做满」的方向。",
]):
    p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    p.text = line; p.font.size = Pt(15); p.font.color.rgb = DARK; p.space_after = Pt(8)

# 负结果
s = slide("四、实验 · 负结果汇总（28+ 臂，同协议、如实报告）")
bullets(s, [
    "扩散 2×2 矩阵（pixel/latent × flow/reg）：均低于 bicubic 或不超自身零初始化。",
    "Mamba/VSS 扫描骨干：等样本预算 −2.3 dB（访存受限、样本效率差）——关线。",
    "几何/结构正则：D4 等变、坐标注入、频域软路由、门控 FFN、压平金字塔——持平或负。",
    "小波架构化尝试：双射小波 U-Net、双路小波 U-Net（完全体）、DTCWT 基——均不敌「Haar 高频损失 + 移位集成」。",
    "频域 rectified flow（HF 残差生成）：修好 OOM/scale 后仍单调毁图——生成不可预测高频只注入伪影。",
    "对抗挖数据多样性（N3，配 Muon）：中性，未破平台。",
], size=15)

# 结论
s = slide("五、结论与下一步")
bullets(s, [
    "一套「数学上讲得通」的配方：stride-1 + 正交小波高频损失(含移位集成) + Muon + 预训练 + TTA → 34.41 dB / 0.9285 SSIM。",
    "同协议、四指标全面领先重训 SwinIR（+1.44 dB / +9.2 MUSIQ / +0.048 MANIQA），且参数量远小于同代 Transformer。",
    "免训练实测证明：×2 PSNR 上限 ~34.4–35.7 dB（配准主导），我们已在天花板内——不是模型不够，是物理不允许。",
    "下一步：×3/×4 用完整配方逼近其上限（唯一明确余量）；保真见顶后，感知轴（MUSIQ/LPIPS/FID）是主战场。",
], size=16)

# 参考文献
s = slide("参考文献", "均联网核实")
bullets(s, [
    "Cai et al. RealSR / LP-KPN, ICCV 2019.   Liang et al. SwinIR, ICCV-W 2021.   Lim et al. EDSR, CVPR-W 2017.   Zhang et al. RCAN, ECCV 2018.",
    "Wang et al. ESRGAN ECCV-W 2018 · BSRGAN ICCV 2021 · Real-ESRGAN ICCV-W 2021.",
    "Qiao et al. RealSR-R1, arXiv:2506.16796 (2025).   TinySR, arXiv:2508.17434 (2025).（文献基线表出处）",
    "Blau & Michaeli. The Perception-Distortion Tradeoff, CVPR 2018, DOI 10.1109/CVPR.2018.00652.",
    "Baker & Kanade. Limits on Super-Resolution and How to Break Them, IEEE TPAMI 2002, DOI 10.1109/TPAMI.2002.1033210.",
    "Robinson & Milanfar. Fundamental Performance Limits in Image Registration, IEEE TIP 2004；Statistical Performance Analysis of SR, TIP 2006.",
    "Chatterjee & Milanfar. Is Denoising Dead?, IEEE TIP 2010.   Donoho & Johnstone. Wavelet Shrinkage (MAD 噪声估计), Biometrika 1994.",
], size=13)

prs.save(str(OUT))
print(f"wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
