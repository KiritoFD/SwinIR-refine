# RealSR / WaveSR 项目长期记忆

**唯一底账：`docs/results/RESULT_LEDGER.md`（156 份 eval.json）**；入口 `README.md`（六阶段脉络）；
算力口径 `docs/INFRA.md`；逐次 run `exp/FINAL_REPORT.md`；被取代的 27 份阶段文档在 `exp/archive/`。
本文件只记「不看文档就会踩坑」的东西。

## 硬件与远程

- RTX 4090 48G，`ds@10.222.120.101`，仓库在 **`~/realsr`**（不是 ~/RealSR），env `harness-qwen`，
  远端 `python` 不在 PATH，要用 conda 绝对路径；`~/realsr` **不是 git 仓库**，结果靠 tar 取回。
- 选 batch 按吞吐不按显存：超 ~40G 呈超线性变慢。latent-S b64 132smp/s > b112 125；
  latent-XS b96 > b320；pixel-S b28 略 > b24；pixel-XS b32 > b64。
- 加速只能上 `torch.compile`：latent-S 1.26×（值），pixel-S 1.05×（只省显存 40.8→20.7G）。

## 达成的结论（背下来）

- **交付**：stride-1 U-Net b64（18.67M）+ 预训练 + DWT λ8 + Muon 5e-3 + 移位 + WD-DCP + TTA
  = **Y 34.4632 / SSIM 0.9284**；单次前向 34.3800；Scale-96(41.19M) 34.5419（只 +0.07）。
- **杠杆排序**：stride-1/native-LR **+0.38** >> 小波+Muon+移位合计 +0.28 > 预训练 +0.09
  > base64→96 +0.07 = WD-DCP +0.07 > TTA +0.06 >> 容量 18.7M→72.5M +0.05。
- **地板**：bicubic 31.7335（旧 README 写 31.7312 是错的）；bicubic 过 VAE 往返 31.5971。
- **感知与保真对立**：最高 MUSIQ/MANIQA 是**原始单次前向**（56.02/0.3534），
  WD-DCP/TTA/大模型全都在拿感知换保真。
- **架构优势是真的**：`unet_ctrl`（骨干 + 纯基线配方，无任何外挂）34.0339 已超 SwinIR-L 33.9707。
- **容量饱和点 ~5M**，三次验证（U-Net b32/40/48、DiT、b128 72.5M、HAT-L 40.7M）。
- **上限是配准墙 34.4–35.7**，不是噪声（49 dB）。
- flow 全线否（pixel 26.79 / latent 28.36 / HF-flow 31.07 且 scale 增大单调崩）；Mamba 否（33.42）。

## 指标口径（错一次就白算）

- 官方：limited-range Y（BT.601 16-235）、uint8、crop=0、shave=0、tile 64/pad 16、100 对均值。
- **参数量只用 `sum(p.numel() for p in net.parameters())`**。数 state_dict 会把
  `relative_position_index` 等 int64 buffer 算进去（SwinIR 14.41M→真值 11.75M，SRFormer 16.37M→10.40M）。
- **MFU 只认 fwd+bwd 口径**；forward-only 系统低估 ~3×。U-Net b64 20.0%、b128 34.7%、DiT-S 30.8%。
- baseline 阶梯各臂共享的是样本**上限** 1.28M，不是同等消耗（HAT-L 0.20M / SwinIR 1.06M），引用必须带注。

## 排障（真踩过）

- **`pkill -f "<脚本名>"` 会自杀**（ssh 的 `bash -c` 命令行含该串）→ 用 `[c]hain_afte` 自排除。
- **`tmux has-session -t NAME` 是前缀匹配**：`baserep` 匹配到 `baserep2` 自己 → 死锁。精确写 `-t '=baserep'`。
- `start_36h.sh` 遇同名 session 打印提示并 `exit 0`；被 kill 的链路会留**死 session** 让后续阶段静默跳过。
- 后台链路用 mkdir 原子锁启动器；单靠 pkill 会留 0 或 2 个实例（两个实例互相 kill-session）。
- **改网络结构必须同步 eval 端构造**：scale96 有 2/4 段评估崩在 `Unexpected key(s): opt_embed.mlp.*`。
- `model.align = in_stride × 2^(levels-1) × out_scale`（=32）；尺寸非 align 倍数要先 pad 再裁回（1000%32=8）。
- `eval_official.py` 经 `| tee` 块缓冲，**别用 tail 判进度**，看进程 + nvidia-smi。
- 评估成本：pixel flow 9–10 s/NFE/对（100 对 @NFE16 ≈ 4 h）；latent flow ~7 min；reg ~12 min。
- Windows 写 `.sh` 必须 LF（`write_text` 会写 CRLF → 远端 syntax error）。git 已加 `.gitattributes` 强制。
- 本机跑 torch 用 `C:\Users\xy\AppData\Local\Programs\Python\Python312\python.exe`（只够 smoke）。
  `python` 里 `glob(os.path.join(ROOT,'a/**/b'))` 不递归，要用 `os.path.join(ROOT,'a','**','b')`。
- 国内：`HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 不通）。

## 课程汇报 PPT（`docs/slides_new/`，13 页，slidep 驱动）

页序 `["01","02","03","11","05","06","07","12","14","13","08","09","10"]`，**页序 ≠ 文件编号**：
`11`=上限（第 4 页）、`12`=存算比（第 8 页）、`14`=infra MFU 大表（第 9 页）、`13`=Mamba 失败（第 10 页）、
`08/09/10`=结论三页（11–13 页）。`04.slide`、`03_tmp.slide` 已删。
流程：`Write slides/NN.slide` → `slidep lint` 必须 `ok:true` → `slidep upsert-dsl --page-index N-1`；
定稿才跑全量 `rebuild.py`（30–130s，带 3 次重试 + `"failures":[]` 校验）。
单页改动不要全量重建，用 upsert + PowerPoint COM 导 PNG 核对（`slidep screenshot` 本机不可用）。

**坑①：横向分栏写死宽度，列宽总和 ≤ 1140px**（1280−70−70）。超了会溢出，且诊断被塞进同一 message
尾部截断成 "and N more"。排查：逐行删后重跑 lint。中文按 1、西文按 0.55 估宽。
**坑②：`<Table>` 不支持 `columnWidths`**（不报错但无效），列宽按内容分配且宽内容列权重高 →
长文案列把数值列挤到折行。唯一手段是**压缩最宽列文案**：4 个指标别塞一格（拆 4 列、用短表头、
靠金色字+深色底分组）、名称列只留名称+参数、表头能省则省、边距 70→56。
**坑③：lint 报 `bottom+Npx` 指向 Table 时，改 Table 的 height 无效**（按行数撑最小高度），必须上移外层 top。
**坑④：右栏卡片组溢出优先压内层 padding/gap/lineHeight**（比移外层 top 省事）。
**坑⑤：副标题折成 2 行就报 bottom 溢出** → 压到一行（≤75 字符）或把进度信息挪进标题括号。

其他硬约束：`Math` 必须 `latex=` + `width` + `display` 且**不能有子元素**（留 20–40px 余量）；
inline `<svg>` 必须显式 width/height；`<Image>` 的 src 相对 PPTX 目录解析**必须绝对路径**；
**emoji 渲染成方框**；写文件必须 `newline='\n'`。
改页码的批量替换**必须用 `</Text>` 做锚点**（否则误改数据里的 `xx.09 / 10.xx`）。

**用户语言规范**：零内部黑话、零比喻词（墙/地板/天花板/抹平）、零营销句式（先说清/换句话说）、
不贴命令行、不要目录页与章节扉页、技术路线按综述口径（CNN→Transformer→GAN→扩散，各自优劣）、
数据优先于叙述、**全篇字号 ≥ 12px**、负面结果可用幽默语气（如「Mamba Out!」）。
