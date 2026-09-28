"""Diff the *stored* configs of two training checkpoints (non-equal flattened keys only).

Usage: python /tmp/ckpt_cfg_diff.py old_ckpt.pth new_ckpt.pth
"""
import sys

import torch
from omegaconf import OmegaConf


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def load_cfg(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("config")
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    return flatten(cfg)


old, new = sys.argv[1], sys.argv[2]
A, B = load_cfg(old), load_cfg(new)
print(f"OLD {old}: {len(A)} config keys")
print(f"NEW {new}: {len(B)} config keys")
print("--- differing keys (model/run/dataset scope) ---")
interesting = ("model.", "run.", "datasets.")
for k in sorted(set(A) | set(B)):
    if not k.startswith(interesting):
        continue
    va, vb = A.get(k, "<absent>"), B.get(k, "<absent>")
    if va != vb:
        print(f"  {k}\n      OLD = {va!r}\n      NEW = {vb!r}")