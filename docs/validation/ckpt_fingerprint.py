"""Fingerprint a training checkpoint: structure, LoRA key count, dtypes, optimizer state.

Usage: python /tmp/ckpt_fingerprint.py ckpt1.pth [ckpt2.pth ...]
"""
import collections
import sys

import torch

for path in sys.argv[1:]:
    print(f"===== {path}")
    try:
        ck = torch.load(path, map_location="cpu")
    except Exception as e:  # noqa: BLE001
        print(f"  !! cannot load: {type(e).__name__}: {e}")
        continue
    print(f"  top-level keys: {list(ck.keys())}")
    print(f"  epoch={ck.get('epoch')} best_agg_metric={ck.get('best_agg_metric')} "
          f"config present={ck.get('config') is not None}")

    sd = ck.get("model", {})
    lora = {k: v for k, v in sd.items() if "lora_" in k}
    pc = {k: v for k, v in sd.items() if k.startswith("pc_encoder")}
    print(f"  model state entries: {len(sd)} | lora keys: {len(lora)} | pc_encoder keys: {len(pc)}")
    print(f"  lora dtype histogram: {dict(collections.Counter(str(v.dtype) for v in lora.values()))}")
    print(f"  pc_encoder dtype histogram: {dict(collections.Counter(str(v.dtype) for v in pc.values()))}")

    st = ck.get("optimizer", {})
    if isinstance(st, dict) and st.get("state"):
        exp = [s.get("exp_avg") for s in st["state"].values() if s.get("exp_avg") is not None]
        print(f"  optimizer: {len(st.get('param_groups', []))} param_groups, "
              f"{len(st['state'])} state entries, exp_avg dtypes="
              f"{dict(collections.Counter(str(t.dtype) for t in exp))}")
        n = [len(g.get("params", [])) for g in st.get("param_groups", [])]
        print(f"  optimizer params per group: {n} (total {sum(n)})")
    else:
        print("  optimizer: none/empty")
    sc = ck.get("scaler", None)
    print(f"  scaler: {type(sc).__name__ if sc is not None else None}"
          + (f" scale={sc.state_dict().get('scale')}" if hasattr(sc, "state_dict") else ""))
    print()