#!/usr/bin/env python3
"""
Export the primitive V1 (PCP-MAE + MaskTransformer) encoder backbone
WITHOUT auto-generating cls_token / cls_pos.

背景说明
--------
统一权重导出流程后，PCP-MAE_with_Objaverse/ckpt_extract.py 会在导出时对
MaskTransformer 自动补齐 cls_token / cls_pos（trunc_normal, std=0.02）。
该行为对 V1-random、Point-MAE+MaskTransformer 是期望的（所有 MaskTransformer
编码器在初始化方式上保持一致），但也因此丢掉了「原始 primitive-V1（无 cls）」
的导出方法。

本脚本复刻统一导出之前被删除的专用导出逻辑：提取 MAE_encoder 骨干并保存为
MiniGPT-3D 格式（{'base_model': ...}），但**不**补齐 cls_token / cls_pos。
用于保持实验完整性、复现 primitive-V1（对应已归档权重
`params_weight/pc_encoder/old/PCPMAE+MaskTransformer.pth`，无 cls）的导出过程。

用法（在 MiniGPT-3D 仓库根目录或任意位置执行）:
    python tools/export_primitive_v1.py \
        --ckpt /data/workspace/PCP-MAE_with_Objaverse/experiments/[V1]PCP‑MAE+MaskTransformer/pretrain/pcpmae_minigpt3d/ckpt-best.pth \
        --out /data/workspace/MiniGPT-3D/params_weight/pc_encoder/PCPMAE+MaskTransformer.pth

导出结果不含 cls_token / cls_pos，如接入 MiniGPT-3D下游，理论上可以运行但会警告。
"""

import argparse
import torch
from collections import OrderedDict


def extract_primitive_v1(ckpt_path, out_path):
    src = torch.load(ckpt_path, map_location='cpu')

    # Locate model state dict
    if 'base_model' in src:
        state_dict = src['base_model']
    elif 'model' in src:
        state_dict = src['model']
    elif 'state_dict' in src:
        state_dict = src['state_dict']
    else:
        raise KeyError(
            f"No recognized model state key found in checkpoint. "
            f"Available keys: {list(src.keys())}"
        )

    for meta_key in ('epoch', 'metrics', 'best_metrics'):
        if meta_key in src:
            print(f"[info] checkpoint {meta_key}: {src[meta_key]}")

    new_state = OrderedDict()
    skipped = []

    for k, v in state_dict.items():
        # Strip DDP 'module.' prefix
        if k.startswith('module.'):
            k = k[7:]

        # Only process MAE_encoder keys (the PointTransformer backbone)
        if not k.startswith('MAE_encoder.'):
            continue

        k_body = k[len('MAE_encoder.'):]

        # --- Map to MiniGPT-3D PointTransformer keys ---
        if k_body.startswith('encoder.'):
            new_state[k_body] = v
        elif k_body.startswith('reduce_dim.'):
            new_state[k_body] = v
        elif k_body.startswith('pos_embed.'):
            new_state[k_body] = v
        elif k_body.startswith('blocks.'):
            new_state[k_body] = v
        elif k_body.startswith('norm.'):
            new_state[k_body] = v
        elif k_body in ('cls_token', 'cls_pos'):
            new_state[k_body] = v
        else:
            skipped.append(k_body)

    # Primitive V1 导出**不**补齐 cls_token / cls_pos（这是与 ckpt_extract.py 的关键差异）
    has_cls = 'cls_token' in new_state and 'cls_pos' in new_state
    print(f"[info] cls_token / cls_pos present: {has_cls}")
    if not has_cls:
        print("       Primitive V1: exporting WITHOUT cls_token / cls_pos "
              "(no auto-generation).")

    print(f"[ok] Extracted {len(new_state)} parameter tensors")
    if skipped:
        print(f"[info] Skipped {len(skipped)} non-backbone keys "
              f"(decoder, pred_head, mask_token, etc.)")

    torch.save({'base_model': new_state}, out_path)
    print(f"[ok] Saved -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export primitive V1 (PCP-MAE + MaskTransformer) backbone WITHOUT cls'
    )
    parser.add_argument('--ckpt', required=True,
                        help='Path to PCP-MAE checkpoint (e.g., ckpt-best.pth)')
    parser.add_argument('--out', required=True,
                        help='Output path for MiniGPT-3D compatible weight file')
    args = parser.parse_args()
    extract_primitive_v1(args.ckpt, args.out)