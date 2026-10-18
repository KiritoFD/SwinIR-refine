# DESIGN.md · RealSR 论文汇报

## 1. 风格

**学术风（命中 design-principle.academic）**——蓝白克制、证据驱动、撑满优先、信息密度高、禁营销/装饰/大留白/彩虹。

## 2. 画布与母版

- 画布 1280×720 (16:9)。左右安全边距 40。
- **页眉 A3 模式**（深蓝顶条 #0E3F8C 高度 8px + 低透明内容区背景）——全篇统一。
- **页脚 D**：左 logo/单位 14px 灰（"RealSR · 真实超分研究 · 2026-09"），右页码 14–16px 深蓝（NN/20）。
- C 区（80–660px）宽 1200，高 580，必须填充 ≥ 80%（除封面/扉页/目录）。

## 3. 色彩（≤ 4 hex + 2 字体）

| 角色 | hex | 用途 |
|---|---|---|
| 主蓝 | `#1E4FA8` | 顶条、强调、深蓝实色卡 |
| 藏蓝 | `#0E3F8C` | 标题、数字、底色 |
| 浅蓝 | `#E8EFF8` / `#F0F5FC` | 卡底、表头浅底 |
| 警示红 | `#D9534F` | 结论金句、负结果警示；单页 ≤3% |
| 数据黄 | `#FFC107` | 巨数锚点（仅列表第 03、11、20 页极少量） |
| 文本 | `#1A2230` / `#4A5568` / `#8B97A8` | 主 / 副 / 辅助 |
| 线框 | `#D6DCE5` | 描边 |
| 背景 | `#FFFFFF` / `#F7F9FC` | 主背景 |

**配色比**：蓝族 35–55%，白底 30–55%，浅蓝 0–35%，红+黄 ≤5%，金色 = 0。禁紫粉玫红、橙绿大色块、彩虹渐变、纯黑大底。

**字体**：思源黑（Source Han Sans / Noto Sans CJK SC）+ Times Serif（仅封面英文 title）。

## 4. 字号（强制下限）

| 层级 | px | 字重 | 用途 |
|---|---|---|---|
| 封面中文主标 | 60–84 | Bold | 封面 |
| 章节大数字 | 200–280 | Bold | 扉页 |
| A 区主标 | 28–36 | Bold | 每页 |
| 卡内标题 | 20–26 | Bold | 卡 |
| KPI 巨数字 | 72–140 | Bold | 数据强调 |
| 正文 | 18–22 | Regular | 段落 |
| 表格 | 14–18 | Regular / Bold | 表体 |
| 注解 / 页脚 | 12–14 | Regular | 来源、页码 |

正文 ≥ 18px（学术风下限 22 优先）。

## 6. 资源全本：`/report/images/`

- `fig_results.png` — 主结果阶梯（已在 assets/）
- `fig_ceiling.png` — 三墙分解（已在 assets/）
- `fig_ablation.png` — 消融 ΔPSNR（已在 assets/）
- `fig_tradeoff.png` — 感知-失真权衡（已在 assets/）
- 自绘 SVG（inline `<svg>`）：
  - `assets/pair_diagram.svg` — Canon/Nikon 双焦配对示意（05）
  - `assets/unet_diagram.svg` — stride-1 U-Net 简笔（12）
  - `assets/tta_d4.svg` — D4 8 重 TTA 示意（12）
- 严禁生图（ImageGen）；学术风禁摄影插画混用。

## 7. 页面角色与版式映射

| 页 | role | layout | 主视觉 |
|---|---|---|---|
| 01 封面 | hero | 全屏视觉+大标题 | KPI 巨数 |
| 02 目录 | supporting | 上下分栏（左右两列） | L3 编号 |
| 03 引言 | hero | 巨型数字+洞察 | 4 KPI 巨字 |
| 04 章节扉页 01 | transition | 巨型数字+标题 | 大数字 01 |
| 05 协议陷阱 | supporting | 左大图+右侧文字 | pair_diagram.svg |
| 06 谱系 | supporting | 2×2 关键词矩阵 | 4 卡 |
| 07 同行对比边界 | supporting | 上表+下结论 | 表 |
| 08 章节扉页 02 | hero | 巨型数字+标题 | 大数字 02 |
| 09 物理墙 1 | supporting | 左公式+右实测表 | 公式+表 |
| 10 物理墙 2/3 | supporting | 左表+右表+底公式 | 表 |
| 11 数学落点 | supporting | 左公式链+右洞察 | 公式链 |
| 12 模型设计 | supporting | 上中下三段 | U-Net+损失+TTA |
| 13 章节扉页 03 | transition | 巨型数字+标题 | 大数字 03 |
| 14 主表 | supporting | 上全表+下金句 | 表 |
| 15 三墙+上限 | supporting | 左大图+右表 | fig_ceiling.png |
| 16 跨尺度 | supporting | 上表+下表 | 表 |
| 17 负结果 1 | supporting | 上表+下结论 | 表 |
| 18 消融 | supporting | 左大图+右表 | fig_ablation.png |
| 19 权衡 | supporting | 左大图+右两卡 | fig_tradeoff.png |
| 20 结论 | hero | 上三栏卡+下复现命令 | 3 卡 |

## 8. 密度与硬闸门

- 内容页 C 区累计高 ≥ 476px（学术风下限）。
- 表格 ≥ 70% C 区 + 结论。
- 数据图表页：主图 ≥ 50% + 洞察 ≥ 100 字 + ≥1 个关键数字。
- 卡片填充率 ≥ 80%。
- 正文页至少复用 3 类组件、4 种文字修辞、3 层内容层次。
- 红色 + 黄色 ≤ 5% 面积。
- 公式、引用、定理、数据全部来自 `exp/FINAL_REPORT.md` / `exp/RELATED_WORK_AND_UPPER_BOUND.md` / `docs/baseline/README.md` 真实条目；禁杜撰。
- 颜色禁紫粉玫红、橙绿大色块、彩虹渐变、纯黑大底。
- 禁 3D / 卡通 / 二次元 / 营销插画。
- 单页最多 1 张深蓝实色卡，全篇 ≤ 3 张。