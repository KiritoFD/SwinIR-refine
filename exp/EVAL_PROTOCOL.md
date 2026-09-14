# 评估协议对照（官方 RealSR 为准）

**日期：** 2026-09-14（本轮按官方 Test.m 精确对齐）

## 官方论文

| 项 | 内容 |
|----|------|
| 论文 | Cai, Zeng, Yong, Cao, Zhang. *Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model*. **ICCV 2019** |
| 仓库 | https://github.com/csjcai/RealSR |
| 数据 | RealSR V3：559 场景（459 train / **100 test**），Canon 5D3 + Nikon D810，HR/LR 分辨率不同 |
| 模型 | Laplacian Pyramid KPN (**LP-KPN**) |
| 另一文 | NTIRE 2019 RealSR Challenge（部分数据 + 比赛协议） |

README 明确：**Trained on the RGB domain and tested on Y channel (Version 3)**。

官方 V3 表只报了 ×3/×4（×2 为 “-”），×2 是社区后续常用设置；本仓库目标是 **V3 ×2**，指标口径仍按官方 Test.m。

## 官方 Test.m 做了什么（本仓库 `model/eval.py` 对齐）

1. 读 RGB → `modcrop(..., 4)`（LR、HR 都按 **4** 对齐，不是按 scale）
2. `rgb2ycbcr` → 取 **Y**（ITU-R BT.601 **limited-range**，Y∈[16,235]）
3. `im2uint8` 后算 PSNR/SSIM（`Cal_PSNRSSIM(..., crop=0)`，**不 shave**）
4. 大图（两边都 >1200）按 1200/stride800 分块；本仓库 V3×2 多数图 ≤1250，默认 tile 推理
5. Canon + Nikon 全量 Test 取 mean
6. RGB PSNR 仅作辅助，**不是**论文主指标

Y 公式（MATLAB `rgb2ycbcr`，RGB∈[0,1]）：

```
Y = 16 + 65.481·R + 128.553·G + 24.966·B
```

## 三套常见协议（不要横比）

| 协议 | 数据 | 倍率 | 退化 | 指标 |
|------|------|------|------|------|
| **官方 RealSR（采用）** | V3 Test Canon+Nikon | ×2/3/4 | 真实拍摄 | **Y limited-range uint8，crop=0** |
| SwinIR 论文 | Set5/14/BSD/Urban | ×2/3/4 | bicubic | Y，常 shave=scale |
| sr-scaling | DIV2K val100 | ×4 | bicubic | RGB 全图，border=scale |

## 本仓库旧数字 vs 官方

旧表用的是 **full-range luma** `0.299R+0.587G+0.114B` + float PSNR + 默认 12 张子集，**不是**官方口径。

- 旧 E11 12 张：RGB 31.73 / full-range Y 32.40（仅内部监控）
- 官方全量数字：以 `python -m model.eval` 输出为准

## 命令

```powershell
python -m model.eval --ckpt experiments\improve\E11_align_loss\ckpt_best.pt `
  --data-root "G:\RealSR\data\RealSR(V3)" --scale 2 --max-pairs 0
```

`--max-pairs 0` = 全量官方 Test（默认）。
