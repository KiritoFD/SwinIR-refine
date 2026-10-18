# 文档地图

**只看一份的话：** 根目录 [`../README.md`](../README.md)（历史脉络 + 方法 + 对比 + 复现）。
**只要数字的话：** [`results/RESULT_LEDGER.md`](results/RESULT_LEDGER.md)（唯一底账）。

## 正式文档（结论以此为准）

| # | 文件 | 管什么 |
|---|---|---|
| 1 | [`../README.md`](../README.md) | 项目入口。六阶段历史脉络、四块方法积木、基线对比、物理上限、负面线路、协议、复现、结构、踩坑 |
| 2 | [`results/RESULT_LEDGER.md`](results/RESULT_LEDGER.md) | **结果总账**。远程 48G 机 156 份 `eval.json` 的全部实测读数，按实验族分组 |
| 3 | [`results/all_evals.csv`](results/all_evals.csv) | 同上，机器可读（`n` 列 = 评测对数，筛 `n==100` 得全量表） |
| 4 | [`INFRA.md`](INFRA.md) | **算力总表**。MFU（统一 fwd+bwd 口径）、显存/batch/形状解析式、数据管线、评估耗时账、远程排程规则 |
| 5 | [`baseline/README.md`](baseline/README.md) | 基线目录：我们用官方原码重训了哪些臂，读数与配方 |
| 6 | [`baseline/SURVEY.md`](baseline/SURVEY.md) | 2021–2026 真实超分方法全景（>40 个），复现 vs 引用的取舍 |
| 7 | [`../exp/FINAL_REPORT.md`](../exp/FINAL_REPORT.md) | 逐次实验编年史（106 次 run 的假设/配置/结论），附 A–F 附录 |
| 8 | [`../exp/METRIC_SUITE_AND_DCP.md`](../exp/METRIC_SUITE_AND_DCP.md) | 八维评测体系 + WD-DCP 的完整推导与证明 |
| 9 | [`../exp/DCP_RESULTS.md`](../exp/DCP_RESULTS.md) | DCP 步长/核参扫描专报 |
| 10 | [`../exp/RELATED_WORK_AND_UPPER_BOUND.md`](../exp/RELATED_WORK_AND_UPPER_BOUND.md) | 相关工作引用 + 三墙上限的测量方法 |

## 汇报材料

| 文件 | 说明 |
|---|---|
| [`slides_new/`](slides_new/) | **现行 13 页汇报**（`RealSR_真实超分_论文汇报.pptx` / `.pdf`）。源文件是 `slides/NN.slide`（slidep DSL），`rebuild.py` 全量重建 |
| [`slides_new/DESIGN.md`](slides_new/DESIGN.md) · [`STORY.md`](slides_new/STORY.md) | 设计约束与叙事结构 |
| [`figs/`](figs/) | 论文级配图（results / ablation / ceiling / tradeoff） |
| [`archive_slidev/`](archive_slidev/) | 旧版 Slidev 幻灯片（14 页，已被 `slides_new` 取代，留作历史） |

## 归档

| 位置 | 内容 |
|---|---|
| [`../exp/archive/`](../exp/archive/README.md) | 27 份阶段文档（SwinIR / 扩散 / Mamba / 小波各期），带分类索引 |
| [`results/logs/`](results/logs/) | 远程机器日志归档：58 份训练/评估日志 + 各臂 `args.json` |

`exp/archive/` 里的文档**不作为结论来源**，但它们记录了失败时的判断，
写论文的 related-work 和复盘时最有用。

## 工具与复现

| 文件 | 用途 |
|---|---|
| [`results/collect_evals.py`](results/collect_evals.py) | 从 `.tmp_remote/` 重新生成 `all_evals.csv` |
| `bench_mfu.py` / `bench_mfu2.py` / `bench_mamba.py` | MFU 实测（fwd+bwd, `FlopCounterMode`）/ Mamba 双后端 |
| [`../diffusion/oracle.py`](../diffusion/oracle.py) | 免训练上限探测：`python -m diffusion.oracle --scale 2\|3\|4` |
| [`../diffusion/bench.py`](../diffusion/bench.py) | 单臂 ms/step、峰值显存（排计划前先跑） |
| [`../scripts/server/`](../scripts/server/) | 全部训练/评测流水线（tmux + 每臂 GPU 等待 + 幂等 `eval.json` 跳过） |
| [`../scripts/test_wd_dcp.py`](../scripts/test_wd_dcp.py) | WD-DCP 高频零损伤单测 |

## 约定

- **指标**：±0.05 dB、感知指标小幅波动 = 平手。SSIM / MUSIQ / MANIQA 与 PSNR 同权。
- **协议**：A-strict（100 对 / limited-range Y / uint8 / modcrop4 / shave 0 / tile 64 pad 16）。
  标签 B-128-512 / C-REdeg / D-blind 的数字**不得进 A-strict 表**。
- **参数量**：只认 `sum(p.numel() for p in net.parameters())`，不数 `state_dict`。
- **MFU**：只认 fwd+bwd 口径，见 `INFRA.md`。
- **一臂一卡**：一个 b64 stride-1 臂就能占满 4090，所有 campaign 串行。
