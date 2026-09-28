"""Compare the *saved model state* key sets of two checkpoints (which params were tracked).

Usage: python /tmp/ckpt_keydiff.py old.pth new.pth
"""
import collections
import sys

import torch


def keys(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck.get("model", {})


def group(kk):
    out = collections.Counter()
    for k in kk:
        if "lora_" in k:
            out["<lora>"] += 1
        elif k.startswith("pc_encoder"):
            out["pc_encoder"] += 1
        else:
            parts = k.split(".")
            out[".".join(parts[:2])] += 1
    return out


a, b = keys(sys.argv[1]), keys(sys.argv[2])
print(f"OLD entries={len(a)}  NEW entries={len(b)}")
ga, gb = group(a), group(b)
print("--- grouped counts (OLD -> NEW) ---")
for k in sorted(set(ga) | set(gb)):
    m = "  <<<" if ga.get(k, 0) != gb.get(k, 0) else ""
    print(f"  {k:35s} {ga.get(k, 0):5d} -> {gb.get(k, 0):5d}{m}")

only_old = sorted(set(a) - set(b))
only_new = sorted(set(b) - set(a))
print(f"--- only in OLD: {len(only_old)} keys, sample ---")
for k in only_old[:8]:
    print("   -", k)
print(f"--- only in NEW: {len(only_new)} keys, sample ---")
for k in only_new[:8]:
    print("   +", k)