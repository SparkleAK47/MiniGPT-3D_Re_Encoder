"""
[已废弃] V1 hybrid 合并脚本 —— 仅供历史复现参考

历史做法：把 Baseline（ULIP-2 / Point-BERT）的 cls_token / cls_pos 拷贝到
V1（PCP-MAE + MaskTransformer）骨干上，得到 V1 hybrid 权重。

当前规范（见 PCP-MAE_with_Objaverse README）：V1 hybrid 方式已弃用，所有
MaskTransformer 编码器统一由 `ckpt_extract.py` 在导出时自动随机初始化
cls_token / cls_pos（trunc_normal, std=0.02），以保证各 MaskTransformer 变体
的 cls 初始化方式一致、消除混淆变量。

本脚本仅保留历史逻辑，默认已不再使用；如需运行请从 MiniGPT-3D 仓库根目录
执行 `python tools/merge_cls_from_baseline.py`，并按需修改其中的输入/输出路径。
"""

import torch

BASELINE = './params_weight/pc_encoder/point_model.pth'
PRIMITIVE_V1 = '/data/workspace/MiniGPT-3D/params_weight/pc_encoder/old/PCPMAE+MaskTransformer.pth'
OUTPUT = './params_weight/pc_encoder/old/PCPMAE+MaskTransformer_hybrid-cls.pth'

original = torch.load(BASELINE, map_location='cpu')
new = torch.load(PRIMITIVE_V1, map_location='cpu')

# 假设原权重也有 base_model 格式，可根据实际调整
if 'base_model' in original:
    new['base_model']['cls_token'] = original['base_model']['cls_token']
    new['base_model']['cls_pos'] = original['base_model']['cls_pos']
    torch.save(new, OUTPUT)
else:
    print("Original model does not contain 'base_model' key. Please check the structure of the original model.")
