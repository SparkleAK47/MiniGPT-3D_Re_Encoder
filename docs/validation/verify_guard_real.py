"""Real-model A/B probe: does the patch actually change the AMP-relevant dtype state?

Run A: force_trainable_fp32 monkeypatched to a no-op  -> reproduces the pre-patch
       MiniGPT_3D state (fp16 trainable adapters) that makes GradScaler.step raise.
Run B: untouched patched code                          -> must end with 0 fp16 trainable.

Run: cd /data/workspace/MiniGPT-3D && PYTHONPATH=. python -u /tmp/verify_guard_real.py <cfg>
"""
import collections
import logging
import sys
import types

import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import minigpt4.models  # noqa: F401
from minigpt4.models.minigpt_v2 import MiniGPT_3D

CFG = sys.argv[1]
cfg = Config(types.SimpleNamespace(cfg_path=CFG, options=[]))

records = []


class Capture(logging.Handler):
    def emit(self, record):
        records.append(record.getMessage())


logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.INFO)


def build():
    return registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)


def stats(model, tag):
    tr = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    hist = collections.Counter(str(p.dtype) for _, p in tr)
    fp16 = [n for n, p in tr if p.dtype == torch.float16]
    print(f"[{tag}] trainable={len(tr)} dtypes={dict(hist)} fp16_trainable={len(fp16)}",
          flush=True)
    if fp16[:3]:
        print(f"    fp16 examples: {fp16[:3]}", flush=True)
    return len(fp16)


print(f"### cfg: {CFG} | amp: {cfg.run_cfg.get('amp')}", flush=True)

# ---- Run A: simulate the pre-patch code by disabling the guard ----
orig = MiniGPT_3D.force_trainable_fp32
MiniGPT_3D.force_trainable_fp32 = lambda self, tag="": []
pre = None
try:
    m = build()
    pre = stats(m, "A PRE-PATCH (guard disabled)")
    del m
finally:
    MiniGPT_3D.force_trainable_fp32 = orig
torch.cuda.empty_cache()

# ---- Run B: the patched code path ----
m = build()
post = stats(m, "B POST-PATCH (guard active)")
casts = [r for r in records if "[amp] cast" in r]
print("cast log lines:", casts, flush=True)
del m

ok = (pre is None or pre > 0) and post == 0
print(f"PRE={pre} POST={post}")
print("RESULT:", "PASS (patch removes fp16 trainables in the real model)" if ok
      else "INCONCLUSIVE/FAIL")
sys.exit(0 if ok else 1)