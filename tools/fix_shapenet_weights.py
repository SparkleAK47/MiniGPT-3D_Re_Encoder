#!/usr/bin/env python3
"""
Fix ShapeNet55-34 exported weight for MiniGPT-3D compatibility.

Two issues to fix:
1. encoder.first_conv.0.weight:  [128, 3, 1]  ->  [128, 6, 1]
   ShapeNet has XYZ only (3ch), MiniGPT-3D expects XYZ+RGB (6ch).
   Solution: pad RGB channels with zeros (since default color is black).
2. Missing cls_token, cls_pos: randomly initialized.

Usage (from MiniGPT-3D repo root):
    python tools/fix_shapenet_weights.py \
        --src params_weight/pc_encoder/old/ShapeNet55-34_3ch.pth \
        --dst params_weight/pc_encoder/old/ShapeNet55-34_6ch+cls.pth
"""

import argparse
import torch
from collections import OrderedDict


def fix_shapenet_weights(src_path, dst_path):
    ckpt = torch.load(src_path, map_location='cpu')
    dst = OrderedDict()

    for k, v in ckpt['base_model'].items():
        # ----- Fix 1: pad first_conv from 3ch to 6ch -----
        if k == 'encoder.first_conv.0.weight' and v.shape[0:2] == (128, 3):
            old_w = v.clone()
            # create new weight with 6 input channels, zero-init
            new_w = torch.zeros(128, 6, 1, dtype=v.dtype, device=v.device)
            new_w[:, :3, :] = old_w                    # copy XYZ channels
            # RGB channels remain zeros → equivalent to black color input
            v = new_w
            print(f'[FIX] {k}: {tuple(old_w.shape)} -> {tuple(new_w.shape)} '
                  f'(XYZ copied, RGB zeroed)')

        dst[k] = v

    # ----- Fix 2: add cls_token and cls_pos if missing -----
    if 'cls_token' not in dst:
        dst['cls_token'] = torch.zeros(1, 1, 384)
        torch.nn.init.trunc_normal_(dst['cls_token'], std=0.02)
        print(f'[FIX] Added cls_token: shape={tuple(dst["cls_token"].shape)} '
              f'(trunc_normal init)')

    if 'cls_pos' not in dst:
        dst['cls_pos'] = torch.randn(1, 1, 384)
        torch.nn.init.trunc_normal_(dst['cls_pos'], std=0.02)
        print(f'[FIX] Added cls_pos:  shape={tuple(dst["cls_pos"].shape)} '
              f'(trunc_normal init)')

    torch.save({'base_model': dst}, dst_path)
    print(f'\nSaved: {len(dst)} tensors -> {dst_path}')

    # Verify
    ref = torch.load(
        '/data/workspace/MiniGPT-3D/params_weight/pc_encoder/point_model.pth',
        map_location='cpu')
    ref_dict = ref['base_model']
    fixed_dict = dst

    missing = set(ref_dict.keys()) - set(fixed_dict.keys())
    unexpected = set(fixed_dict.keys()) - set(ref_dict.keys())
    mismatches = [(k, tuple(fixed_dict[k].shape), tuple(ref_dict[k].shape))
                  for k in (set(ref_dict.keys()) & set(fixed_dict.keys()))
                  if ref_dict[k].shape != fixed_dict[k].shape]

    print('\n--- Verification ---')
    if not missing:
        print('✅ No missing keys')
    else:
        print(f'❌ Missing: {missing}')
    if not unexpected:
        print('✅ No unexpected keys')
    else:
        print(f'⚠️  Unexpected: {unexpected}')
    if not mismatches:
        print('✅ All shapes match')
    else:
        print(f'❌ Shape mismatches: {mismatches}')

    if not missing and not mismatches:
        print('\n✅ Weights are now fully compatible with MiniGPT-3D!')
    else:
        print('\n❌ Still has issues — check above')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', default='params_weight/pc_encoder/old/ShapeNet55-34_3ch.pth')
    parser.add_argument('--dst', default='params_weight/pc_encoder/old/ShapeNet55-34_6ch+cls.pth')
    args = parser.parse_args()
    fix_shapenet_weights(args.src, args.dst)
