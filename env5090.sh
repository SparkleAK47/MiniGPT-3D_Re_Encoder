#!/bin/bash
# ============================================================================
# MiniGPT-3D 环境安装脚本 —— RTX 5090 Dv2 (Blackwell) + CUDA 12.8 + PyTorch 2.7.1
# 环境名: minigpt5090
# 使用方式: 按步骤逐条手动执行，不要一次性 source 运行
# ============================================================================

set -e

echo "================================================================"
echo "   MiniGPT-3D 环境安装 (minigpt5090 / RTX 5090 / CUDA 12.8)"
echo "================================================================"
echo ""
echo "【重要】请按以下步骤逐条执行，每执行完一条确认无误后再继续。"
echo ""

# ============================================================================
# Step 1: 创建 conda 环境 (Python 3.10)
# ============================================================================
cat << 'STEP1'
--------------------- Step 1: 创建 conda 环境 ---------------------
执行以下命令:
  conda create -n minigpt5090 python=3.10 -y
  conda activate minigpt5090
STEP1

read -p "Step 1 完成后按回车继续..."

# ============================================================================
# Step 2: 安装 PyTorch 2.7.1 + CUDA 12.8 (手动执行，下载约 3GB)
# ============================================================================
cat << 'STEP2'
--------------------- Step 2: 安装 PyTorch 2.7.1 + CUDA 12.8 ---------------------
【手动执行，耗时约 5-10 分钟】

  pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

  # 验证安装:
  python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

【注】triton 使用 PyTorch 内置的 pytorch-triton，无需单独安装。
STEP2

read -p "Step 2 完成后按回车继续..."

# ============================================================================
# Step 3: 核心 ML 库
# ============================================================================
cat << 'STEP3'
--------------------- Step 3: 核心 ML 库 (transformers, accelerate, peft, bitsandbytes) ---------------------
STEP3

pip install transformers==4.51.0
pip install accelerate==1.7.0
pip install bitsandbytes==0.45.3
pip install peft==0.15.0
pip install safetensors==0.5.0
pip install tokenizers==0.21.0
pip install sentencepiece==0.2.0

read -p "Step 3 完成后按回车继续..."

# ============================================================================
# Step 4: 科学计算 & 图像处理
# ============================================================================
cat << 'STEP4'
--------------------- Step 4: 科学计算 & 图像处理 ---------------------
STEP4

pip install numpy==1.26.2
pip install pandas==2.1.4
pip install scipy==1.11.4
pip install scikit-learn==1.3.2
pip install scikit-image==0.22.0
pip install matplotlib==3.7.0
pip install pillow==10.1.0
pip install opencv-python==4.7.0.72
pip install imageio==2.33.1
pip install tifffile==2023.12.9
pip install packaging==23.2

read -p "Step 4 完成后按回车继续..."

# ============================================================================
# Step 5: 3D 点云库
# ============================================================================
cat << 'STEP5'
--------------------- Step 5: 3D 点云相关库 ---------------------
STEP5

pip install open3d==0.18.0
pip install plyfile==1.0.3
pip install pyquaternion==0.9.9

read -p "Step 5 完成后按回车继续..."

# ============================================================================
# Step 6: Web / UI 框架
# ============================================================================
cat << 'STEP6'
--------------------- Step 6: Web / UI 框架 ---------------------
STEP6

pip install gradio==4.44.0
pip install fastapi==0.105.0
pip install uvicorn==0.24.0.post1
pip install dash==2.16.1
pip install flask==3.0.2
pip install starlette==0.27.0
pip install python-multipart==0.0.9

read -p "Step 6 完成后按回车继续..."

# ============================================================================
# Step 7: HuggingFace 生态
# ============================================================================
cat << 'STEP7'
--------------------- Step 7: HuggingFace 生态 ---------------------
STEP7

pip install datasets==2.15.0
pip install huggingface-hub==0.19.4
pip install sentence-transformers==2.2.2
pip install nltk==3.8.1

read -p "Step 7 完成后按回车继续..."

# ============================================================================
# Step 8: 训练 / 实验管理
# ============================================================================
cat << 'STEP8'
--------------------- Step 8: 训练 / 实验管理 ---------------------
STEP8

pip install wandb==0.16.1
pip install tensorboard
pip install tqdm==4.64.1
pip install loguru==0.7.2
pip install rich==13.7.0

read -p "Step 8 完成后按回车继续..."

# ============================================================================
# Step 9: 模型组件库
# ============================================================================
cat << 'STEP9'
--------------------- Step 9: 模型组件库 ---------------------
STEP9

pip install einops==0.7.0
pip install timm==0.6.13
pip install omegaconf==2.3.0
pip install addict==2.4.0
pip install easydict==1.11

read -p "Step 9 完成后按回车继续..."

# ============================================================================
# Step 10: 数据处理 & 存储
# ============================================================================
cat << 'STEP10'
--------------------- Step 10: 数据处理 & 存储 ---------------------
STEP10

pip install pyyaml==6.0
pip install orjson==3.9.10
pip install pyarrow==14.0.2
pip install xxhash==3.4.1
pip install objaverse==0.1.7
pip install webdataset==0.2.48
pip install decord==0.6.0
pip install iopath==0.1.10
pip install fvcore
pip install yacs

read -p "Step 10 完成后按回车继续..."

# ============================================================================
# Step 11: 框架/工具
# ============================================================================
cat << 'STEP11'
--------------------- Step 11: 框架 / 工具 ---------------------
STEP11

pip install requests==2.31.0
pip install aiohttp==3.9.1
pip install aiofiles==23.2.1
pip install openai==0.28.1
pip install dashscope==1.20.4
pip install termcolor==2.4.0
pip install colorama
pip install configargparse==1.7
pip install psutil==5.9.4
pip install gputil==1.4.0
pip install progressbar2==4.3.0
pip install retrying==1.3.4
pip install portalocker==2.8.2
pip install sentry-sdk==1.39.1

read -p "Step 11 完成后按回车继续..."

# ============================================================================
# Step 12: Jupyter / Notebook 支持
# ============================================================================
cat << 'STEP12'
--------------------- Step 12: Jupyter / Notebook 支持 ---------------------
STEP12

pip install ipython==8.18.1
pip install ipykernel==6.27.1
pip install jupyter-client==8.6.0
pip install jupyter-core==5.5.1
pip install ipywidgets==8.1.2

read -p "Step 12 完成后按回车继续..."

# ============================================================================
# Step 13: 创建数据目录 & 解压 NLTK 数据
# ============================================================================
cat << 'STEP13'
--------------------- Step 13: 创建目录 & NLTK 数据 ---------------------
STEP13

mkdir -p ./data/anno_data ./data/modelnet40_data ./data/objaverse_data
mkdir -p "$HOME/nltk_data/corpora"
unzip -o wordnet.zip -d "$HOME/nltk_data/corpora/"
pip install "setuptools==69.5.1"

STEP13

read -p "Step 13 完成后按回车继续..."

# 其他

pip install transformers==4.36.2
pip uninstall -y transformers peft accelerate tokenizers huggingface-hub
pip install transformers==4.36.2 \
            peft==0.7.1 \
            accelerate==0.25.0 \
            tokenizers==0.15.2 \
            huggingface-hub==0.19.4
cp /data/workspace/MiniGPT-3D/modeling_phi.py \
   $CONDA_PREFIX/lib/python3.10/site-packages/transformers/models/phi/


# ============================================================================
# Step 14: 配置验证 & Phi 模型说明
# ============================================================================
cat << 'PHI_NOTES'

============================================================================
[重要] Phi 模型说明
============================================================================

本项目的 Phi-2 权重(.params_weight/Phi_2/)是旧版 Microsoft 格式，
对应的模型文件是 minigpt4/models/modeling_phi.py（base_model_prefix = "transformer"）。

请确认 minigpt4/models/base_model.py 中的导入路径为:
    from minigpt4.models.modeling_phi import PhiForCausalLM

**根目录的 modeling_phi.py 是给旧版 env_install.sh 覆盖 transformers 用的，
   新环境中不要使用它。**

需要手动修改的文件 (已全部完成，请确认):
  1. [OK] minigpt4/models/configuration_phi.py — 添加新版属性别名
  2. [OK] minigpt4/models/base_model.py — 导入路径已改为本地路径，peft import 已清理

============================================================================
验证环境:
  conda activate minigpt5090
  python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}, PyTorch: {torch.__version__}')"
  python -c "import transformers; print(f'transformers: {transformers.__version__}')"
  python -c "from minigpt4.models.modeling_phi import PhiForCausalLM; print('PhiForCausalLM 导入成功')"

============================================================================

PHI_NOTES

echo ""
echo "===== env5090.sh 所有步骤已完成 ====="
echo "请确认上方验证命令都通过后，运行训练:"
echo "  conda activate minigpt5090"
echo "  cd /data/workspace/MiniGPT-3D"
echo "  CUDA_VISIBLE_DEVICES=0 python train.py --cfg-path ./train_configs/MiniGPT_3D/stage_1.yaml"