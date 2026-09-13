# Mod-SwinIR / RealSR V3 Pipeline

针对 RealSR V3（真实配准 + 复杂相机退化）的现代化 SwinIR 训练管线。

## 环境

- Python: `C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe`
- GPU: RTX 4070 Laptop (~8GB)
- 数据: `data/RealSR(V3)/{Canon,Nikon}/{Train,Test}/{2,3,4}/`

## 架构（ModSwinIR）

1. Shallow 3x3 Conv
2. **RAPE** 径向位置编码（镜头中心-边缘感知）
3. **DeformAlign** 可变形对齐（缓解 1–3px 配准误差）
4. 6× RSTB：**OCA**（重叠窗口注意力）→ **GDFN**（门控深度卷积 FFN）→ **AMF**（傅里叶振幅调制）
5. PixelShuffle 重建 + 像素级 **uncertainty** log-variance 图
6. **UWCL** 损失：不确定性加权 L1 + 轻量 Contextual + 梯度项

## 快速验证

```powershell
# 组件 + 端到端 smoke
python -m mod_swinir.smoke_test

# 30 step / 2 对图 smoke
python -m mod_swinir.train --smoke --scale 2

# 小 batch 真实训练（默认 batch=2, LR patch=48, 峰值显存 ~0.25GB）
python -m mod_swinir.train --scale 2 --batch-size 2 --lr-patch 48 --steps 200 --amp
```

## 常用参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `--scale` | 2 | 本次训练的放大倍数（2/3/4） |
| `--batch-size` | 2 | 小显存建议 2 |
| `--lr-patch` | 48 | LR 裁块边长（HR=48*scale） |
| `--steps` | 200 | 迭代步数 |
| `--amp` | off | 混合精度（AMF/FFT 已强制 fp32） |
| `--grad-accum` | 1 | 梯度累积，可等效更大 batch |
| `--max-train-pairs` | all | 子采样训练对数 |

## 输出

- `experiments/mod_swinir_x2/ckpt_last.pt`
- `experiments/mod_swinir_x2/preview_step*.pt`（LR_bicubic | SR | HR 横向拼接张量）

## 显存参考（本机实测）

| 配置 | 峰值显存 |
|------|----------|
| batch=2, LR48, x2, 无 AMP | ~0.25 GB |
| batch=2, LR48, x2, AMP | ~0.20 GB |

模型约 **0.76M** 参数（tiny 配置：embed=48, 6×RSTB depth=1）。
