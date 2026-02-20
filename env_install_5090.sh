#!/bin/bash

cp modeling_phi.py "$CONDA_PREFIX/lib/python3.9/site-packages/transformers/models/phi/"
mkdir -p "$HOME/nltk_data/corpora"
unzip wordnet.zip
cp -r wordnet "$HOME/nltk_data/corpora"

mkdir -p ./data/anno_data
mkdir -p ./data/modelnet40_data
mkdir -p ./data/objaverse_data

set -e

source /opt/miniconda3/bin/activate activate minigpt_3d

pip uninstall torch torchvision torchaudio triton
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 triton==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install huggingface-hub==0.23.0
pip install -U hf-transfer