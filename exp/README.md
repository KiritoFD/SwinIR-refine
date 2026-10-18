# 实验档案（exp/）

这里只放**仍有结论价值**的实验文档。已经被取代的阶段文档在
[`archive/`](archive/README.md)，数字底账在
[`../docs/results/RESULT_LEDGER.md`](../docs/results/RESULT_LEDGER.md)。

| 文件 | 内容 | 状态 |
|---|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | 逐次实验编年史：106 次 run 的假设 / 配置 / 结果 / 判定，附 A–F 附录（总表、root 清单、逐 run 配置台账、审计发现、上限探测） | 现行 |
| [`METRIC_SUITE_AND_DCP.md`](METRIC_SUITE_AND_DCP.md) | 八维评测体系定义 + WD-DCP 的完整推导、证明与实现细节 | 现行 |
| [`DCP_RESULTS.md`](DCP_RESULTS.md) | DCP 步长 η、核参 σ、自适应色差扫描的全部读数 | 现行 |
| [`RELATED_WORK_AND_UPPER_BOUND.md`](RELATED_WORK_AND_UPPER_BOUND.md) | 相关工作（带核实的引用）+ 三墙上限的测量方法与跨尺度结果 | 现行 |
| [`OFFICIAL_EVAL.md`](OFFICIAL_EVAL.md) | 官方全量评估的协议与早期读数 | 现行（协议部分） |
| [`RESULTS.md`](RESULTS.md) | 早期结果表与命令 | 部分过时，数字以底账为准 |

## 归档文档的定位

`archive/` 里有 27 份文档，按阶段分类。它们**不作为结论来源**，
但记录了失败时的判断依据 —— 写 related-work、复盘「为什么当时会走那条路」时最有用。
`archive/README.md` 给了每份文档一句话说明和「与今天结论的关系」。

## 阅读顺序建议

1. 想知道发生了什么 → [`../README.md`](../README.md) 第一节（六阶段）
2. 想要某个数字 → [`../docs/results/RESULT_LEDGER.md`](../docs/results/RESULT_LEDGER.md)
3. 想知道某次 run 为什么这么设 → [`FINAL_REPORT.md`](FINAL_REPORT.md) 附录 D（逐 run 配置台账）
4. 想引用相关工作 → [`RELATED_WORK_AND_UPPER_BOUND.md`](RELATED_WORK_AND_UPPER_BOUND.md)
5. 想复现 → [`../README.md`](../README.md) 第七节
