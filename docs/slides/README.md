# RealSR 报告幻灯片（Slidev）

中文学术结构：**引言 → 相关工作 → 方法（数学）→ 实验（密集大表）**。源文件 `slides.md`，配图在 `public/figs/`。

## 运行 / 导出
```bash
npm install          # 安装 slidev
npm run dev          # 本地预览（默认 http://localhost:3030）
npm run build        # 构建为静态站点 -> dist/
npm run export       # 导出 PDF / PPTX（需 playwright-chromium，已列入 devDeps）
# 导出为 pptx：
npx slidev export --format pptx
# 导出为 pdf：
npx slidev export --format pdf
```

## 重新生成配图（可选，改动数据后）
```bash
python ../../docs/make_figs.py           # 生成 docs/figs/*.png
cp ../../docs/figs/fig_*.png public/figs/
```

## 结构一览
| 段 | 页 |
|---|---|
| 封面 / 大纲 | 1–2 |
| 一、引言 | 问题本质 · 误差分解与理论上限 |
| 二、相关工作 | 谱系表 · 本工作定位 |
| 三、方法 | 总体形式 · 正交小波高频损失 · 移位集成/DTCWT · Muon · 预训练+TTA |
| 四、实验 | 主表①(A-strict) · 主表②(文献同协议,全方法全指标) · 上限实测 · 权衡 · 消融 · 跨尺度 · 负结果 · 配图 |
| 五、结论 + 参考文献 | |

数学用 KaTeX（`$...$` / `$$...$$`）；表格最优加粗、次优下划线；协议标签严守（跨协议不横比）。
