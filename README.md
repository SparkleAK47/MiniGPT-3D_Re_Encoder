# MiniGPT-3D — 复现与点云编码器替代研究

本项目基于 [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) 官方实现，在成功复现全部四阶段训练流程后，进行了**低成本自监督点云编码器替代 ULIP-2 预训练 Point-BERT** 的系统研究。核心候选方案为 **PCP-MAE** 及其变体（V1 / V2 / Point-MAE），辅以 ShapeNet55-34 对照实验与随机权重消融。

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
├── PCP-MAE/                    # ShapeNet55-34 预训练分支（MaskTransformer）
│   ├── cfgs/pretrain/base.yaml           # ShapeNet55-34 预训练配置
│   ├── models/pointbert_mg/              # MiniGPT-3D 兼容的 PointTransformer 实现
│   ├── ckpt_extract.py                   # 通用权重提取脚本
│   └── README.md
│
├── PCP-MAE_with_Objaverse/     # Objaverse 660K 预训练分支（V1/V2/Point-MAE）
│   ├── cfgs/pretrain/
│   │   ├── base.yaml                     # V1 配置 (MaskTransformer)
│   │   ├── base_minigpt_encoder.yaml     # V2 配置 (PointTransformer)
│   │   └── ablation_point_mae.yaml       # Point-MAE 消融配置
│   ├── models/PCP_MAE.py                 # 模型定义（含 PointTransformerMAEEncoder）
│   ├── models/pointbert_mg/              # 从 MiniGPT-3D 复制的 PointTransformer 实现
│   ├── ckpt_extract.py                   # 通用权重提取脚本（V2/Point-MAE 适用）
│   ├── tools/export_minigpt_encoder.py   # V2 专用导出脚本
│   ├── ckpt-extract_for_pcpmae-pretrain.py  # V1 手动提取脚本
│   └── README.md
│
└── MiniGPT-3D/                 # 下游训练与评测中心（本仓库）
    ├── params_weight/pc_encoder/         # 所有编码器权重存放处
    ├── train_configs/MiniGPT_3D/         # 四阶段训练配置（stage_1~5.yaml）
    ├── eval_configs/                     # 评测配置
    ├── pointllm/eval/                    # 评测脚本
    ├── minigpt4/models/pointbert/        # 下游使用的 PointTransformer 定义
    ├── fix_pcpmae_shapenet.py            # ShapeNet 权重修复（3ch→6ch + 补 cls）
    ├── hybrid.py                         # V1 hybrid 合并（补 cls）
    ├── random_weight.sh                  # 随机权重生成
    ├── point_model_VS_hybrid.py          # 编码器特征诊断对比脚本
    ├── check_weights.py                  # 检查 Phi-2 嵌入一致性
    └── README.md                         # 本文件
```

---

## 2. 编码器变体一览

| 代号 | 权重文件 | 预训练方式 | 预训练数据 | 架构 | cls 来源 |
|------|----------|------------|------------|------|----------|
| **Baseline** | `point_model.pth` | ULIP-2 / Point-BERT（官方） | Objaverse | PointTransformer (self-attn) | 原生 |
| **V1** | `point_model_pcpmae.pth` | PCP-MAE + MaskTransformer | Objaverse 660K | MaskTransformer (cross-attn) | **无 cls** |
| **V1 hybrid** | `point_model_hybrid.pth` | V1 骨干 + Baseline cls | Objaverse 660K | MaskTransformer | 从 Baseline 拷贝 |
| **V2** | `point_model_pcp_v2.pth` | PCP-MAE + PointTransformer | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 |
| **Point-MAE** | `point_model_pointmae.pth` | 纯 Point-MAE（ita=0） | Objaverse 660K | PointTransformer (self-attn) | 预训练原生 |
| **ShapeNet55-34** | `pcpmae_ShapeNet_fixed.pth` | PCP-MAE + MaskTransformer | ShapeNet55-34 | MaskTransformer (cross-attn) | 随机初始化（修复补充） |
| **Random** | `point_model_random.pth` | 随机初始化（无预训练） | — | PointTransformer (self-attn) | 随机初始化 |

### 架构差异说明

- **MaskTransformer (V1 / ShapeNet55-34)**：使用 cross-attention（visible + mask 双分支），预训练时无 cls token。与 MiniGPT-3D 下游推理使用的标准 PointTransformer（self-attention + cls token）实现不同，因此需要额外步骤补充 cls 参数。
- **PointTransformer (V2 / Point-MAE / Random)**：预训练直接采用与 MiniGPT-3D 完全一致的 PointTransformer 结构（self-attention + cls token），权重可无缝导出，无需额外合并。

### 实验对照关系

| 对比维度 | 实验组 |
|----------|--------|
| 预训练数据域 | Objaverse (V1/V2) vs ShapeNet55-34 |
| 编码器架构 | MaskTransformer (V1) vs PointTransformer (V2) |
| 预训练任务 | PCP-MAE (V2) vs Point-MAE (ita=0) |
| 预训练知识增益 | V2 vs Random (随机初始化) |
| 冻结 vs 解冻 | 每个变体均做了 `freeze_pc: True` 与 `freeze_pc: False`（stage_5）对比 |

---

## 3. 前置准备

### 3.1 环境

```bash
conda create -n minigpt_3d python=3.10 -y
conda activate minigpt_3d

# PyTorch
conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia

# 其余依赖
cd /data/workspace/MiniGPT-3D
pip install -r environment.yml  # 或手动安装
```

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

编码器预训练在 `/data/workspace/PCP-MAE_with_Objaverse/`（Objaverse）或 `/data/workspace/PCP-MAE/`（ShapeNet55-34）中进行。详细训练命令参考对应仓库的 README。

### 4.1 V1：PCP-MAE + MaskTransformer（Objaverse）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base.yaml \
  --exp_name pcpmae_minigpt3d \
  --seed 42
```

输出目录：`experiments/base/pretrain/pcpmae_minigpt3d/`

### 4.2 V2：PCP-MAE + PointTransformer（Objaverse）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base_minigpt_encoder.yaml \
  --exp_name pcp_minigpt_encoder_objaverse \
  --seed 42
```

输出目录：`experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/`

### 4.3 Point-MAE 消融（Objaverse）

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/ablation_point_mae.yaml \
  --exp_name point_mae_objaverse \
  --seed 42
```

输出目录：`experiments/ablation_point_mae/pretrain/point_mae_objaverse/`

### 4.4 ShapeNet55-34 预训练

```bash
cd /data/workspace/PCP-MAE

CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/pretrain/base.yaml \
  --exp_name pcpmae_pretrain
```

输出目录：`experiments/base/pretrain/pcpmae_pretrain/`

---

## 5. 权重导出与修复（接入 MiniGPT-3D）

预训练后的 checkpoint 需要提取编码器骨干权重，并转换为 MiniGPT-3D 兼容的格式（`{'base_model': {...}}`）。目标存放路径统一为：

```
/data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

### 5.1 V1 权重导出（手动，无 cls）

V1 使用 MaskTransformer，预训练 checkpoint 中无 `cls_token`/`cls_pos`。

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

# 手动提取（参考 ckpt-extract_for_pcpmae-pretrain.py 的逻辑）
python ckpt-extract_for_pcpmae-pretrain.py
# 输出：point_model_pcpmae.pth

# 复制到 MiniGPT-3D
cp point_model_pcpmae.pth /data/workspace/MiniGPT-3D/params_weight/pc_encoder/
```

### 5.2 V1 hybrid 合并（补充 cls）

将 Baseline 的 `cls_token` / `cls_pos` 并入 V1 骨干：

```bash
cd /data/workspace/MiniGPT-3D

python hybrid.py
# 输出：point_model_hybrid.pth（在 MiniGPT-3D 根目录）
# 注意：hybrid.py 中的路径可能需要根据实际情况调整
```

`hybrid.py` 的逻辑：
```python
new['base_model']['cls_token'] = original['base_model']['cls_token']
new['base_model']['cls_pos'] = original['base_model']['cls_pos']
```

### 5.3 V2 / Point-MAE 权重导出（含 cls）

V2 和 Point-MAE 使用与 MiniGPT-3D 一致的 PointTransformer，checkpoint 中已包含 `cls_token`/`cls_pos`。

```bash
cd /data/workspace/PCP-MAE_with_Objaverse

# V2：专用导出脚本
python tools/export_minigpt_encoder.py \
  --pcp-ckpt experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth

# Point-MAE：通用提取脚本
python ckpt_extract.py \
  --ckpt experiments/ablation_point_mae/pretrain/point_mae_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pointmae.pth
```

### 5.4 ShapeNet55-34 权重修复

ShapeNet 数据仅 3 维 xyz，而 MiniGPT-3D 期望 6 维 xyz+rgb，且预训练为 MaskTransformer（无 cls）。需使用 `fix_pcpmae_shapenet.py` 修复：

```bash
cd /data/workspace/MiniGPT-3D

python fix_pcpmae_shapenet.py \
  --src params_weight/pc_encoder/pcpmae_ShapeNet.pth \
  --dst params_weight/pc_encoder/pcpmae_ShapeNet_fixed.pth
```

修复内容：
1. `encoder.first_conv.0.weight`：`[128, 3, 1]` → `[128, 6, 1]`（RGB 通道补零，等价于黑色输入）
2. 补充 `cls_token` / `cls_pos`（trunc_normal 随机初始化）

### 5.5 随机权重生成

```bash
cd /data/workspace/MiniGPT-3D

bash random_weight.sh
# 输出：params_weight/pc_encoder/point_model_random.pth
```

### 5.6 权重导出脚本对照

| 脚本 | 位置 | 适用编码器 | 输出格式 |
|------|------|-----------|----------|
| `tools/export_minigpt_encoder.py` | PCP-MAE_with_Objaverse | V2 | 含 cls 的 `base_model` |
| `ckpt_extract.py` | PCP-MAE / PCP-MAE_with_Objaverse | V2、Point-MAE | 含 cls，带完整性校验 |
| `ckpt-extract_for_pcpmae-pretrain.py` | PCP-MAE_with_Objaverse | V1（手动提取） | 无 cls |
| `hybrid.py` | MiniGPT-3D | V1 → V1 hybrid | 补充 cls |
| `fix_pcpmae_shapenet.py` | MiniGPT-3D | ShapeNet55-34 | 3ch→6ch + 补 cls |
| `random_weight.sh` | MiniGPT-3D | Random | 随机初始化 |

---

## 6. MiniGPT-3D 四阶段训练

### 6.1 修改训练配置

在 `train_configs/MiniGPT_3D/stage_{1,2,3,4}.yaml`（以及可选的 `stage_5.yaml`）中设置 `pc_encoder_ckpt` 和 `freeze_pc`：

```yaml
model:
  pc_encoder_ckpt: "./params_weight/pc_encoder/point_model_pcp_v2.pth"  # 替换为你的权重
  freeze_pc: True    # 冻结编码器（默认）
  # freeze_pc: False  # 解冻编码器（stage_5 实验）
```

各实验对应配置示例：

| 实验 | `pc_encoder_ckpt` | `freeze_pc` |
|------|-------------------|-------------|
| Baseline | `point_model.pth` | `True` |
| V1 hybrid | `point_model_hybrid.pth` | `True` / `False` |
| V2 | `point_model_pcp_v2.pth` | `True` / `False` |
| Point-MAE | `point_model_pointmae.pth` | `True` |
| ShapeNet55-34 | `pcpmae_ShapeNet_fixed.pth` | `True` / `False` |
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

---

## 7. 评估

### 7.1 修改评测配置

在 `eval_configs/benchmark_evaluation_paper.yaml` 中设置 checkpoint 路径和编码器权重：

```yaml
model:
  ckpt: '/data/workspace/MiniGPT-3D/output/<实验名>/stage_3/checkpoint_2.pth'
  second_ckpt: "/data/workspace/MiniGPT-3D/output/<实验名>/stage_4/checkpoint_0.pth"
  # 若所有 stage 均为 freeze_pc: True，必须指定 pc_encoder_ckpt
  pc_encoder_ckpt: "./params_weight/pc_encoder/point_model_pointmae.pth"
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

python point_model_VS_hybrid.py \
  --ckpt-a ./params_weight/pc_encoder/point_model.pth \
  --ckpt-b ./params_weight/pc_encoder/point_model_pcp_v2.pth \
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
python check_weights.py
```

---

## 9. 工具脚本索引

### 权重提取与修复

| 文件 | 位置 | 用途 |
|------|------|------|
| `fix_pcpmae_shapenet.py` | MiniGPT-3D | ShapeNet 权重修复（3ch→6ch + 补 cls） |
| `hybrid.py` | MiniGPT-3D | V1 权重合并 cls（从 Baseline 拷贝） |
| `random_weight.sh` | MiniGPT-3D | 生成随机初始化 PointTransformer 权重 |
| `ckpt_extract.py` | PCP-MAE / PCP-MAE_with_Objaverse | 通用权重提取（V2/Point-MAE 适用） |
| `tools/export_minigpt_encoder.py` | PCP-MAE_with_Objaverse | V2 专用导出脚本 |
| `ckpt-extract_for_pcpmae-pretrain.py` | PCP-MAE_with_Objaverse | V1 手动提取参考脚本 |

### 诊断与评测

| 文件 | 位置 | 用途 |
|------|------|------|
| `point_model_VS_hybrid.py` | MiniGPT-3D | 两个编码器特征对比（cosine、kNN、intra-inter） |
| `check_weights.py` | MiniGPT-3D | 检查 Phi-2 嵌入是否与官方一致 |
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
├── point_model.pth              # Baseline（ULIP-2 / Point-BERT 官方）
├── point_model_pcpmae.pth       # V1（原始，无 cls）
├── point_model_hybrid.pth       # V1 hybrid（由 hybrid.py 生成，含 cls）
├── point_model_pcp_v2.pth       # V2（PointTransformer + PCP-MAE）
├── point_model_pointmae.pth     # Point-MAE 消融
├── pcpmae_ShapeNet.pth          # ShapeNet55-34（原始，需修复）
├── pcpmae_ShapeNet_fixed.pth    # ShapeNet55-34（修复后）
└── point_model_random.pth       # 随机初始化消融
```

### 10.2 预训练 checkpoint 存档

预训练的完整 checkpoint 已备份至：

```
/data/datasets/PCP_checkpoint/
├── Objaverse_bad_loss/          # V1 训练早期（loss 异常高）
├── Objaverse_good_loss/         # V1 训练稳定后
├── Objaverse_v2/                # V2 预训练 checkpoint
├── official/                    # 官方 PCP-MAE 预训练权重
└── ShapeNet55-34/               # ShapeNet55-34 预训练 checkpoint
```

### 10.3 MiniGPT-3D 训练输出

已完成的实验输出：

```
/data/datasets/MiniGPT-3D/
├── first-checkpoint/            # 首次成功复现
├── first-pcpmae/                # 首次 PCP-MAE 接入
├── hybrid-with-objaverse/       # V1 实验
├── hybrid-with-objaverse_unfreeze/  # V1 解冻实验
├── pointmae/                    # Point-MAE 实验
├── random_unfreeze/             # 随机权重解冻消融
├── ShapeNet/                    # ShapeNet55-34 冻结实验
├── ShapeNet_unfreeze/           # ShapeNet55-34 解冻实验
├── v2/                          # V2 冻结实验
└── v2_unfreeze/                 # V2 解冻实验
```

### 10.4 权重复制命令参考

```bash
# 从存档恢复权重到 MiniGPT-3D（示例）
cp /data/datasets/PCP_checkpoint/Objaverse_v2/ckpt-last.pth \
   /data/workspace/PCP-MAE_with_Objaverse/experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth

# 导出 V2 权重
cd /data/workspace/PCP-MAE_with_Objaverse
python tools/export_minigpt_encoder.py \
  --pcp-ckpt experiments/base_minigpt_encoder/pretrain/pcp_minigpt_encoder_objaverse/ckpt-last.pth \
  --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model_pcp_v2.pth
```

---

## 11. 环境与踩坑记录

- **CUDA 兼容性**：RTX 3090 下使用 PyTorch 2.0.1 + CUDA 11.8；RTX 5090 尝试了多个 CUDA 版本均存在兼容性问题，最终放弃。
- **C++ 扩展编译**：`chamfer_dist` 和 `emd` 扩展在 PyTorch 2.x 下可能需要手动修复编译错误。
- **AMP / bfloat16**：预训练中开启 bfloat16 AMP 可节省显存，但需注意 NaN 问题（尤其在 Chamfer 距离计算中）。
- **BLIP2 权重**：官方硬编码 URL 不可访问，需手动下载后修改 `minigpt_v2.py` 中的 `load_from_pretrained` 路径。
- **Qwen API**：原 Qwen2-72B-Instruct 已下线，已适配 Qwen-Flash（阿里云百炼）。
- **单卡训练**：PCP-MAE 预训练通过 `step_per_update` 累积梯度实现等效大 batch，单卡 RTX 3090 可训练。

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>

本项目基于 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>。

本仓库所有修改与新增内容在同一许可证下发布。

## Acknowledgements

- [MiniGPT-3D](https://github.com/TangYuan96/MiniGPT-3D) — 原始项目
- [PCP-MAE](https://github.com/aHapBean/PCP-MAE) — 编码器预训练方法
- [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) — 点云 MAE 基线
