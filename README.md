# MiniGPT-3D — 复现与点云编码器替代研究

本项目基于 [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) 官方实现，在成功复现全部四阶段训练流程后，进行了**低成本自监督点云编码器替代 ULIP-2 预训练 Point-BERT** 的系统研究。核心候选方案为 **PCP-MAE** 及其变体（V1-random / V2 / Point-MAE×2），辅以 ShapeNet55-34 对照实验与随机权重消融。

---

## 目录

- [1. 项目结构总览](#1-项目结构总览)
- [2. 编码器变体一览](#2-编码器变体一览)
- [3. 前置准备](#3-前置准备)
- [4. 编码器预训练（PCP-MAE 侧）](#4-编码器预训练pcp-mae-侧)
- [5. 权重导出与修复（接入 MiniGPT-3D）](#5-权重导出与修复接入-minigpt-3d)
- [6. MiniGPT-3D 四阶段训练](#6-minigpt-3d-四阶段训练)
- [7. 评估](#7-评估)
- [8. 编码器质量诊断](#8-编码器质量诊断)
- [9. 工具脚本索引](#9-工具脚本索引)
- [10. 权重文件管理](#10-权重文件管理)
- [11. 环境与踩坑记录](#11-环境与踩坑记录)

---

## 1. 项目结构总览

三个关联仓库的关系：

```
/data/workspace/
├── PCP-MAE_with_ShapeNet/        # ShapeNet55-34 预训练分支（MaskTransformer）
│   ├── cfgs/pretrain/[ShapeNet55-34]PCP‑MAE+MaskTransformer.yaml
│   ├── models/pointbert_mg/              # MiniGPT-3D 兼容的 PointTransformer 实现
│   ├── ckpt_extract.py                   # 通用权重提取脚本（自动补 cls）
│   └── README.md
│
├── PCP-MAE_with_Objaverse/       # Objaverse 660K 预训练分支（V1/V2/Point-MAE×2）
│   ├── cfgs/pretrain/
│   │   ├── [V1]PCP‑MAE+MaskTransformer.yaml   # V1（MaskTransformer + PCP-MAE）
│   │   ├── [V2]PCP‑MAE+PointTransformer.yaml  # V2（PointTransformer + PCP-MAE）
│   │   ├── Point‑MAE+PointTransformer.yaml    # Point-MAE 消融（PointTransformer，ita=0）
│   │   └── Point‑MAE+MaskTransformer.yaml     # Point-MAE 消融（MaskTransformer，ita=0）
│   ├── models/PCP_MAE.py                   # 模型定义（含 PointTransformerMAEEncoder + MaskTransformer）
│   ├── ckpt_extract.py                     # 通用权重提取脚本（自动补 cls）
│   └── README.md
│
└── MiniGPT-3D/                   # 下游训练与评测中心（本仓库）
    ├── params_weight/pc_encoder/           # 所有编码器权重存放处（当前权重 + old/ 归档）
    ├── train_configs/MiniGPT_3D/           # 四阶段训练配置（stage_1~4.yaml）
    ├── eval_configs/                       # 评测配置
    ├── pointllm/eval/                      # 评测脚本
    ├── minigpt4/models/pointbert/          # 下游使用的 PointTransformer 定义
    ├── tools/                              # 工具脚本（权重导出/修复/诊断，见 §9）
    │   ├── export_primitive_v1.py          # primitive-V1（无 cls）导出
    │   ├── fix_shapenet_weights.py         # ShapeNet 权重修复（3ch→6ch + 补 cls）
    │   ├── merge_cls_from_baseline.py      # [已废弃] V1 hybrid 合并（补 cls）
    │   ├── compare_encoders_features.py    # 编码器特征诊断对比
    │   └── check_phi2_embedding.py         # 检查 Phi-2 嵌入一致性
    ├── random_weight.sh                    # 随机权重生成
    └── README.md                           # 本文件
```

---

## 2. 编码器变体一览

| 代号 | 权重文件 | 预训练方式 | 预训练数据 | 架构 | cls 来源 | dtype |
|------|----------|------------|------------|------|----------|-------|
| **Baseline** | `point_model.pth` | ULIP-2 / Point-BERT（官方） | Objaverse | PointTransformer (self-attn) | 原生 | **float16** |
| **V1-random** | `point_model_V1+random-cls.pth` | PCP-MAE + MaskTransformer | Objaverse 660K | MaskTransformer (cross-attn) | 随机初始化（自动补） | float32 |
| **V2** | `point_model_v2.pth` | PCP-MAE + PointTransformer | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 | float32 |
| **Point-MAE** | `old/PointMAE+PointTransformer.pth` | 纯 Point-MAE（ita=0） | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 | float32 |
| **Point-MAE+MaskTransformer** | `old/PointMAE+MaskTransformer+cls.pth` | 纯 Point-MAE（ita=0）+ MaskTransformer | Objaverse 660K | MaskTransformer (cross-attn) | 随机初始化（自动补） | float32 |
| **ShapeNet55-34** | `old/ShapeNet55-34_6ch+cls.pth` | PCP-MAE + MaskTransformer | ShapeNet55-34 | MaskTransformer (cross-attn) | 随机初始化（修复补充） | float32 |
| **Random** | `point_model_random.pth` | 随机初始化（无预训练） | — | PointTransformer (self-attn) | 随机初始化 | float32 |

> **dtype 说明**：只有 Baseline `point_model.pth`（官方 ULIP-2）以 **float16** 存储（158 个 fp16 + 2 个 int64 tensor）；其余编码器权重均为 **float32**。

### 已废弃 / 归档变体

| 代号 | 权重文件（`old/`） | 废弃原因 |
|------|--------------------|----------|
| V1（原始，无 cls） | `PCPMAE+MaskTransformer.pth` | 被 **V1-random** 取代（cls 补充方式统一为随机初始化） |
| V1 hybrid | `PCPMAE+MaskTransformer_hybrid-cls.pth` | 从 Baseline 拷贝 cls，与「随机初始化 cls」混淆变量，已弃用 |
| V2 旧导出名 | `PCPMAE+PointTransformer_old-name.pth` | 与 V2 现行导出 `point_model_v2.pth` 重复 |
| Point-MAE+MaskTransformer（无 cls 旧导出） | `PointMAE+MaskTransformer.pth` | 统一导出时已自动补 cls，新名 `PointMAE+MaskTransformer+cls.pth` |

### 架构差异说明

- **MaskTransformer（V1-random / Point-MAE+MaskTransformer / ShapeNet55-34）**：使用 cross-attention（visible + mask 双分支），预训练时无 cls token。与 MiniGPT-3D 下游推理使用的标准 PointTransformer（self-attention + cls token）实现不同，因此导出时需要额外补 cls 参数（由 `ckpt_extract.py` 自动随机初始化 `cls_token`/`cls_pos`）。
- **PointTransformer（V2 / Point-MAE / Random）**：预训练直接采用与 MiniGPT-3D 完全一致的 PointTransformer 结构（self-attention + cls token），权重可无缝导出，无需额外合并。

### 实验对照关系

| 对比维度 | 实验组 |
|----------|--------|
| 预训练数据域 | Objaverse (V1-random / V2 / Point-MAE×2) vs ShapeNet55-34 |
| 编码器架构 | MaskTransformer (V1-random) vs PointTransformer (V2) |
| 预训练任务 | PCP-MAE (V2) vs Point-MAE (ita=0) |
| 预训练知识增益 | V2 vs Random (随机初始化) |
| 冻结 vs 解冻 | 每个变体均做了 `freeze_pc: True` 与 `freeze_pc: False` 对比（见 §6.1） |

---

## 3. 前置准备

### 3.1 环境

训练显卡已迁移至 **RTX 5090（Blackwell）**，MiniGPT-3D 下游训练/评测使用 conda 环境 **`minigpt5090`**（Python 3.10，PyTorch 2.7.1 + CUDA 12.8）。分步安装脚本见 [`env5090.sh`](env5090.sh)（请逐条复制命令执行，**不要**直接 `bash` 整个脚本）。

```bash
conda create -n minigpt5090 python=3.10 -y
conda activate minigpt5090

# PyTorch 2.7.1 + CUDA 12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

# 其余依赖参考 env5090.sh（transformers 4.36.2 / peft 0.7.1 等）
cd /data/workspace/MiniGPT-3D
```

> 历史环境 `minigpt_3d`（RTX 3090）安装步骤保留如下，供复现旧实验参考：
>
> ```bash
> conda create -n minigpt_3d python=3.10 -y
> conda activate minigpt_3d
>
> # PyTorch（RTX 3090）
> conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
>
> # 其余依赖
> cd /data/workspace/MiniGPT-3D
> pip install -r environment.yml  # 或手动安装
> ```

### 3.2 权重下载

官方硬编码的在线权重路径已失效，需手动下载以下文件：

1. **BLIP2 预训练权重**：从 [Google Storage](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth) 下载，修改 `minigpt4/models/minigpt_v2.py` 中的 `load_from_pretrained(url_or_filename=...)` 指向本地路径。

2. **Phi-2 模型**：放置于 `./params_weight/Phi_2/`。

3. **MiniGPT-3D 官方 stage_3 checkpoint**：`./params_weight/TinyGPT_V_stage_3/TinyGPT-V_for_Stage3.pth`（用于 stage_1 训练的初始化）。

4. **官方 Point-BERT 权重**：`./params_weight/pc_encoder/point_model.pth`（Baseline 编码器）。

5. **预训练数据集**：
   - Objaverse 点云（`.npy` 格式，8192 点 × 6 维）：`./data/objaverse_data/`
   - ModelNet40 测试数据：`./data/modelnet40_data/modelnet40_test_8192pts_fps.dat`

### 3.3 API 适配

原项目使用的 Qwen2-72B-Instruct API 已下线，已在 `pointllm/eval/evaluator_opensource_llm_QwenAPI.py` 中适配阿里云百炼 **Qwen-Flash** 模型。评估时使用参数 `--model_type qwen-flash`。

---

## 4. 编码器预训练（PCP-MAE 侧）

编码器预训练在 `/data/workspace/PCP-MAE_with_Objaverse/`（Objaverse）或 `/data/workspace/PCP-MAE_with_ShapeNet/`（ShapeNet55-34）中进行。详细训练命令参考对应仓库的 README。

### 4.1 V1-random：PCP-MAE + MaskTransformer（Objaverse）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[V1]PCP‑MAE+MaskTransformer.yaml" \
  --exp_name pcpmae_minigpt3d \
  --seed 42
```

输出目录：`experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/`

### 4.2 V2：PCP-MAE + PointTransformer（Objaverse）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[V2]PCP‑MAE+PointTransformer.yaml" \
  --exp_name pcp_minigpt_encoder_objaverse \
  --seed 42
```

输出目录：`experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/`

### 4.3 Point-MAE 消融（Objaverse，PointTransformer 或 MaskTransformer）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

# Point-MAE + PointTransformer
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/Point‑MAE+PointTransformer.yaml" \
  --exp_name point_mae_objaverse \
  --seed 42

# Point-MAE + MaskTransformer
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/Point‑MAE+MaskTransformer.yaml" \
  --exp_name mask_mae_objaverse \
  --seed 42
```

输出目录：`experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/`、`experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/`

### 4.4 ShapeNet55-34 预训练

```bash
cd /data/workspace/PCP-MAE_with_ShapeNet

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config "cfgs/pretrain/[ShapeNet55-34]PCP‑MAE+MaskTransformer.yaml" \
  --exp_name pcpmae_pretrain
```

输出目录：`experiments/[ShapeNet55-34]PCP‑MAE+MaskTransformer/pretrain/pcpmae_pretrain/`

---

## 5. 权重导出与修复（接入 MiniGPT-3D）

预训练后的 checkpoint 需要提取编码器骨干权重，并转换为 MiniGPT-3D 兼容的格式（`{'base_model': {...}}`）。目标存放路径统一为：

```
/data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

> 统一导出脚本为 `PCP-MAE_with_Objaverse/ckpt_extract.py`（`PCP-MAE_with_ShapeNet` 同名脚本）——支持全部编码器类型，并在导出 MaskTransformer 骨干时**自动补齐**缺失的 `cls_token`/`cls_pos`（trunc_normal，std=0.02）。

### 5.1 V1-random 权重导出（自动补 cls）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

python ckpt_extract.py \
  --ckpt "experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth" \
  --out point_model_V1+random-cls.pth

cp "point_model_V1+random-cls.pth" /data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

### 5.2 V2 权重导出（含原生 cls）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

python ckpt_extract.py \
  --ckpt "experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/ckpt-best.pth" \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth
```

### 5.3 Point-MAE 权重导出

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

# Point-MAE + PointTransformer（含原生 cls）
python ckpt_extract.py \
  --ckpt "experiments/Point‑MAE+PointTransformer/pretrain/point_mae_objaverse/ckpt-best.pth" \
  --out PointMAE+PointTransformer.pth

# Point-MAE + MaskTransformer（自动补 cls）
python ckpt_extract.py \
  --ckpt "experiments/Point‑MAE+MaskTransformer/pretrain/mask_mae_objaverse/ckpt-best.pth" \
  --out PointMAE+MaskTransformer+cls.pth

# 导出后归档（旧名 → 新名，仅供参考）：
#   point_model_pointmae.pth  → PointMAE+PointTransformer.pth
#   point_model_maskmae.pth   → PointMAE+MaskTransformer+cls.pth
#   point_model_pcpmae.pth    → PCPMAE+MaskTransformer.pth
#   point_model_pcp_v2.pth    → PCPMAE+PointTransformer_old-name.pth
#   point_model_hybrid.pth    → PCPMAE+MaskTransformer_hybrid-cls.pth（已废弃）
#   pcpmae_ShapeNet.pth       → ShapeNet55-34_3ch.pth
#   pcpmae_ShapeNet_fixed.pth → ShapeNet55-34_6ch+cls.pth
```

### 5.4 primitive-V1 导出（无 cls，保持实验完整性）

如需复现「primitive-V1（无 cls）」的历史导出（与 `ckpt_extract.py` 自动补 cls 的行为相反），使用 MiniGPT-3D 侧的 `tools/export_primitive_v1.py`：

```bash
cd /data/workspace/MiniGPT-3D

python tools/export_primitive_v1.py \
  --ckpt "/data/workspace/PCP-MAE_with_Objaverse/experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth" \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/PCPMAE+MaskTransformer.pth
```

### 5.5 ShapeNet55-34 权重修复

ShapeNet 数据仅 3 维 xyz，而 MiniGPT-3D 期望 6 维 xyz+rgb。需使用 `tools/fix_shapenet_weights.py` 修复（`ckpt_extract.py` 已在导出时自动补 cls，此步主要是 3ch→6ch）：

```bash
cd /data/workspace/MiniGPT-3D

python tools/fix_shapenet_weights.py \
  --src params_weight/pc_encoder/old/ShapeNet55-34_3ch.pth \
  --dst params_weight/pc_encoder/old/ShapeNet55-34_6ch+cls.pth
```

修复内容：
1. `encoder.first_conv.0.weight`：`[128, 3, 1]` → `[128, 6, 1]`（RGB 通道补零，等价于黑色输入）
2. 补充 `cls_token` / `cls_pos`（trunc_normal 随机初始化，若无）

### 5.6 随机权重生成

```bash
cd /data/workspace/MiniGPT-3D

bash random_weight.sh
# 输出：params_weight/pc_encoder/point_model_random.pth
```

### 5.7 权重导出脚本对照

| 脚本 | 位置 | 适用编码器 | 输出格式 |
|------|------|-----------|----------|
| `ckpt_extract.py` | PCP-MAE_with_Objaverse / PCP-MAE_with_ShapeNet | 全部编码器 | 含 cls（自动补），带完整性校验 |
| `tools/export_primitive_v1.py` | MiniGPT-3D | primitive-V1 | **无 cls**（保持原始） |
| `tools/fix_shapenet_weights.py` | MiniGPT-3D | ShapeNet55-34 | 3ch→6ch + 补 cls |
| `tools/merge_cls_from_baseline.py` | MiniGPT-3D | [已废弃] V1 hybrid | 从 Baseline 拷贝 cls |
| `random_weight.sh` | MiniGPT-3D | Random | 随机初始化 |

---

## 6. MiniGPT-3D 四阶段训练

### 6.1 修改训练配置

在 `train_configs/MiniGPT_3D/stage_{1,2,3,4}.yaml` 中设置 `pc_encoder_ckpt` 和 `freeze_pc`：

```yaml
model:
  pc_encoder_ckpt: "./params_weight/pc_encoder/point_model_v2.pth"  # 替换为你的权重
  freeze_pc: True    # 冻结编码器
  # freeze_pc: False  # 解冻编码器
```

> **关于 `train_pc_encoder` vs `freeze_pc`**：如需解冻编码器，**只需把 `freeze_pc` 改成 `False`**，无需修改 `train_pc_encoder`。`train_pc_encoder` 字段名容易产生歧义，但它不是控制编码器是否参与训练的关键开关；实际决定编码器是否随训的是 `freeze_pc`。

各实验对应配置示例（5 个候选编码器均已通过 freeze 与 unfreeze 两种训练验证；表中 `True` / `False` 为该变体被验证过的两种取值）：

| 实验 | `pc_encoder_ckpt` | `freeze_pc` |
|------|-------------------|-------------|
| Baseline | `point_model.pth` | `True` / `False` |
| V1-random | `point_model_V1+random-cls.pth` | `True` / `False` |
| V2 | `point_model_v2.pth` | `True` / `False` |
| Point-MAE | `old/PointMAE+PointTransformer.pth` | `True` / `False` |
| Point-MAE+MaskTransformer | `old/PointMAE+MaskTransformer+cls.pth` | `True`（已评估） |
| ShapeNet55-34 | `old/ShapeNet55-34_6ch+cls.pth` | `True` / `False` |
| Random | `point_model_random.pth` | `False`（必须解冻） |

### 6.2 启动训练

```bash
cd /data/workspace/MiniGPT-3D

export PYTHONPATH=$PWD
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:${CONDA_PREFIX}/lib:$LD_LIBRARY_PATH
export WANDB_MODE=disabled

# 四阶段顺序执行
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_1.yaml > log_stage_1.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_2.yaml > log_stage_2.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_3.yaml > log_stage_3.txt
CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_4.yaml > log_stage_4.txt
```

训练日志位于项目根目录（`log_stage_*.txt` 或自定义重定向文件），各 stage 输出目录由 yaml 中的 `run.output_dir` 指定（例如 `./output/pcpmae/stage_{1,2,3,4}/`）。

### 6.3 训练配置说明

| Stage | 主要作用 | 关键配置 |
|-------|----------|----------|
| Stage 1 | 训练 pc_linear 投影层 + Q-Former 对齐 | `only_train_pc_linear: True` |
| Stage 2 | 端到端多任务训练 | LoRA 微调 LLM |
| Stage 3 | 进一步微调 | 训练更多 epoch |
| Stage 4 | 最终微调 | 最终权重用于评测 |

> **Freeze / Unfreeze 验证状态**：当前每个编码器（V1-random、V2、Point-MAE、Point-MAE+MaskTransformer、ShapeNet55-34）均已通过 freeze 与 unfreeze 两种训练与评估。其中 **V1-random** 已完成 freeze 与 unfreeze 训练和评估，但评估完毕后其 `output/` 下的权重**未及时改名并粘贴 log 文件**，导致后续训练输出将其覆盖；**Point-MAE+MaskTransformer** 已完成评估，因此其「无 cls」旧导出（`old/PointMAE+MaskTransformer.pth`）无需再次评估。

---

## 7. 评估

### 7.1 修改评测配置

在 `eval_configs/benchmark_evaluation_paper.yaml` 中设置 checkpoint 路径和编码器权重：

```yaml
model:
  ckpt: '/data/workspace/MiniGPT-3D/output/<实验名>/stage_3/checkpoint_2.pth'
  second_ckpt: "/data/workspace/MiniGPT-3D/output/<实验名>/stage_4/checkpoint_0.pth"
  # 若所有 stage 均为 freeze_pc: True，必须指定 pc_encoder_ckpt
  pc_encoder_ckpt: "./params_weight/pc_encoder/point_model_V1+random-cls.pth"
```

> **注意**：若训练时 `freeze_pc: False`（解冻），checkpoint 中已包含 pc_encoder 权重，`pc_encoder_ckpt` 可留空或被 checkpoint 覆盖。若训练时 `freeze_pc: True`（冻结），评测配置中**必须**指定 `pc_encoder_ckpt`。

### 7.2 开放词汇分类（Objaverse）

```bash
cd /data/workspace/MiniGPT-3D

# Prompt 0
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
  --out_path ./evaluate/<实验名> \
  --task_type classification \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 0

# Prompt 1
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
  --out_path ./evaluate/<实验名> \
  --task_type classification \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 1
```

### 7.3 ModelNet40 闭集分类

```bash
# Prompt 0
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_modelnet_cls.py \
  --out_path ./evaluate/<实验名> \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 0

# Prompt 1
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_modelnet_cls.py \
  --out_path ./evaluate/<实验名> \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 1
```

### 7.4 物体描述生成

```bash
CUDA_VISIBLE_DEVICES=0 python pointllm/eval/eval_objaverse.py \
  --out_path ./evaluate/<实验名> \
  --task_type captioning \
  --cfg-path ./eval_configs/benchmark_evaluation_paper.yaml \
  --prompt_index 2
```

### 7.5 Qwen-Flash API 主观打分

```bash
export PYTHONPATH=$PWD

# 开放词汇分类
python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py \
  --results_path ./evaluate/<实验名>/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_classification_prompt0.json \
  --eval_type open-free-form-classification \
  --model_type qwen-flash \
  --parallel --num_workers 4

# ModelNet40 闭集分类
python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py \
  --results_path ./evaluate/<实验名>/evaluation/ModelNet_classification_prompt0.json \
  --eval_type modelnet-close-set-classification \
  --model_type qwen-flash \
  --parallel --num_workers 4

# 物体描述
python ./pointllm/eval/evaluator_opensource_llm_QwenAPI.py \
  --results_path ./evaluate/<实验名>/evaluation/PointLLM_brief_description_val_200_GT_Objaverse_captioning_prompt2.json \
  --eval_type object-captioning \
  --model_type qwen-flash \
  --parallel --num_workers 4
```

### 7.6 传统指标评估（BLEU、ROUGE 等）

```bash
python pointllm/eval/traditional_evaluator.py \
  --results_path ./evaluate/<实验名>/evaluation/<结果json>
```

---

## 8. 编码器质量诊断

在投入四阶段训练之前，可以用特征对比脚本快速诊断编码器权重质量。

### 8.1 对比两个编码器的特征空间

```bash
cd /data/workspace/MiniGPT-3D

python tools/compare_encoders_features.py \
  --ckpt-a ./params_weight/pc_encoder/point_model.pth \
  --ckpt-b ./params_weight/pc_encoder/point_model_v2.pth \
  --data-path ./data/modelnet40_data/modelnet40_test_8192pts_fps.dat \
  --max-samples 2468
```

关注指标：
| 指标 | 含义 | 健康范围 |
|------|------|----------|
| Same-sample cosine (A vs B) | 两个编码器同一样本特征的相似度 | cls/router 接近 1.0 |
| kNN accuracy (ModelNet40) | 编码器特征的最近邻分类准确率 | Baseline router ~75%；接近 ~2.5%（随机）则有问题 |
| Intra − Inter margin | 类内 vs 类间余弦相似度差距 | Baseline cls margin ~0.11；接近 0 说明类间不可分 |

### 8.2 检查 Phi-2 嵌入一致性

```bash
cd /data/workspace/MiniGPT-3D
python tools/check_phi2_embedding.py
```

---

## 9. 工具脚本索引

所有 MiniGPT-3D 侧的工具脚本已统一收敛到 `tools/` 目录（避免根目录散落、命名歧义）。

### 权重提取与修复

| 文件 | 位置 | 用途 |
|------|------|------|
| `ckpt_extract.py` | PCP-MAE_with_ShapeNet / PCP-MAE_with_Objaverse | 通用权重提取（自动补 cls，见 §5） |
| `tools/export_primitive_v1.py` | MiniGPT-3D | primitive-V1（无 cls）导出 |
| `tools/fix_shapenet_weights.py` | MiniGPT-3D | ShapeNet 权重修复（3ch→6ch + 补 cls） |
| `tools/merge_cls_from_baseline.py` | MiniGPT-3D | [已废弃] V1 hybrid 合并 cls（从 Baseline 拷贝） |
| `random_weight.sh` | MiniGPT-3D | 生成随机初始化 PointTransformer 权重 |

### 诊断与评测

| 文件 | 位置 | 用途 |
|------|------|------|
| `tools/compare_encoders_features.py` | MiniGPT-3D | 两个编码器特征对比（cosine、kNN、intra-inter） |
| `tools/check_phi2_embedding.py` | MiniGPT-3D | 检查 Phi-2 嵌入是否与官方一致 |
| `note.sh` | MiniGPT-3D | 评估命令速查 |

### 训练与评估

| 文件 | 位置 | 用途 |
|------|------|------|
| `train.py` | MiniGPT-3D | 训练入口 |
| `pointllm/eval/eval_objaverse.py` | MiniGPT-3D | Objaverse 分类/描述评估 |
| `pointllm/eval/eval_modelnet_cls.py` | MiniGPT-3D | ModelNet40 分类评估 |
| `pointllm/eval/evaluator_opensource_llm_QwenAPI.py` | MiniGPT-3D | Qwen API 主观打分 |
| `pointllm/eval/traditional_evaluator.py` | MiniGPT-3D | 传统指标评估（BLEU 等） |

---

## 10. 权重文件管理

### 10.1 MiniGPT-3D 编码器权重目录

```
params_weight/pc_encoder/
├── point_model.pth                   # Baseline（ULIP-2 / Point-BERT 官方，float16）
├── point_model_V1+random-cls.pth     # V1-random（PCP-MAE + MaskTransformer + 随机 cls，float32）
├── point_model_v2.pth                # V2（PCP-MAE + PointTransformer，float32）
├── point_model_random.pth            # Random（随机初始化消融，float32）
└── old/                              # 已归档 / 废弃 / 历史导出权重（详见 §2）
    ├── PointMAE+MaskTransformer.pth        # Point-MAE+MaskTransformer（无 cls 旧导出）
    ├── PointMAE+MaskTransformer+cls.pth    # Point-MAE+MaskTransformer（含 cls，已评估）
    ├── PointMAE+PointTransformer.pth       # Point-MAE+PointTransformer
    ├── PCPMAE+MaskTransformer.pth          # primitive-V1（无 cls）
    ├── PCPMAE+MaskTransformer_hybrid-cls.pth   # V1 hybrid（已废弃）
    ├── PCPMAE+PointTransformer_old-name.pth    # V2 旧导出名
    ├── ShapeNet55-34_3ch.pth           # ShapeNet55-34（修复前，3ch）
    └── ShapeNet55-34_6ch+cls.pth       # ShapeNet55-34（修复后，6ch + cls）
```

> **dtype 规律**：Baseline `point_model.pth` 为 **float16**；其余当前 / 归档权重均为 **float32**（MaskTransformer 的 `cls_token`/`cls_pos` 由随机初始化生成）。

### 10.2 预训练 checkpoint 存档

预训练 checkpoint 与日志统一归档在 `/data/datasets/PCP_checkpoint/`，按训练策略分层（每个预训练变体一个子目录）：

```
/data/datasets/PCP_checkpoint/
├── Standard CosineAnnealing/          # 最终采用的训练策略（无早停 + 余弦退火）
│   ├── [V1]PCP‑MAE+MaskTransformer/     # 含 ckpt-best.pth（V1-random 的最终使用权重）
│   └── [V2]PCP‑MAE+PointTransformer/    # 含 ckpt-best.pth（V2 的最终使用权重）
├── Early-Stopping/                    # 历史（早停）策略，5 个变体各一个子目录
├── official/                          # 官方 PCP-MAE 预训练权重（PCP-MAE-275.pth / PCP-MAE-300.pth）
└── Redundant training tests/          # 冗余训练测试（保留备查，勿作实验依据）
```

### 10.3 MiniGPT-3D 训练输出

下游四阶段训练的输出统一归档在 `/data/datasets/MiniGPT-3D/<实验名>/`，每个实验目录下包含 `stage_1/ … stage_4/` 及对应 `log.txt`：

```
/data/datasets/MiniGPT-3D/
├── baseline_retrain/            # Baseline 重练
├── baseline_unfreeze/           # Baseline 解冻
├── PointMAE-PointTransformer/   # Point-MAE + PointTransformer 冻结
├── PointMAE-Point_unfreeze/     # Point-MAE + PointTransformer 解冻
├── primitive-V1/                # 历史 primitive-V1（无 cls）
├── random_unfreeze/             # Random 解冻消融
├── ShapeNet/                    # ShapeNet55-34 冻结
├── ShapeNet_unfreeze/           # ShapeNet55-34 解冻
├── V1-hybrid/                   # 历史 V1 hybrid（已废弃）
├── V1-hybrid_unfreeze/          # 历史 V1 hybrid 解冻（已废弃）
├── V1-random(pcp-mask)/         # V1-random 冻结
├── v2/                          # V2 冻结
└── v2_unfreeze/                 # V2 解冻
```

> **注意**：V1-random 的评估已完成，但评估完毕后其 `output/` 权重未及时改名并粘贴 log，后续训练输出覆盖了原目录（详见 §6.3）。

### 10.4 权重复制命令参考

```bash
# 从存档恢复 V1/V2 最终使用权重（ckpt-best.pth）
cp "/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V1]PCP‑MAE+MaskTransformer/ckpt-best.pth" \
   "/data/workspace/PCP-MAE_with_Objaverse/experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth"

cp "/data/datasets/PCP_checkpoint/Standard CosineAnnealing/[V2]PCP‑MAE+PointTransformer/ckpt-best.pth" \
   "/data/workspace/PCP-MAE_with_Objaverse/experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/ckpt-best.pth"

# 导出 V2 权重
cd /data/workspace/PCP-MAE_with_Objaverse
python ckpt_extract.py \
  --ckpt "experiments/[V2]PCP‑MAE+PointTransformer/pretrain/pcp_minigpt_encoder_objaverse/ckpt-best.pth" \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_v2.pth
```

---

## 11. 环境与踩坑记录

- **CUDA / 显卡**：已从 RTX 3090（`minigpt_3d` / `pcpmae`，PyTorch 2.0.1 + CUDA 11.8）迁移至 **RTX 5090（Blackwell，sm_120）**，MiniGPT-3D 下游使用 **`minigpt5090`**（PyTorch 2.7.1 + CUDA 12.8），PCP-MAE 预训练使用 **`mae5090`**（PyTorch 2.13.0 + cu130）。RTX 5090 需要 PyTorch ≥2.7（Blackwell 支持），无需再尝试低版本 CUDA 组合。
- **C++ 扩展编译**：`chamfer_dist` / `emd` 在 PyTorch 2.x 下可能需要手动修复编译错误；`mae5090` 环境已改用纯 PyTorch 实现，**无需编译**（详见 PCP-MAE_with_Objaverse README §1.1）。
- **AMP / bfloat16**：预训练中开启 bfloat16 AMP 可节省显存，但需注意 NaN 问题（尤其在 Chamfer 距离计算中）。
- **BLIP2 权重**：官方硬编码 URL 不可访问，需手动下载后修改 `minigpt_v2.py` 中的 `load_from_pretrained` 路径。
- **Qwen API**：原 Qwen2-72B-Instruct 已下线，已适配 Qwen-Flash（阿里云百炼）。
- **单卡训练**：PCP-MAE 预训练通过 `step_per_update` 累积梯度实现等效大 batch；RTX 5090（24GB）+ bf16 可支撑有效 batch=64。

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>

本项目基于 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>。

本仓库所有修改与新增内容在同一许可证下发布。

## Acknowledgements

- [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) — 原始项目
- [PCP-MAE](https://github.com/aHapBean/PCP-MAE) — 编码器预训练方法
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) — 点云 MAE 基线
