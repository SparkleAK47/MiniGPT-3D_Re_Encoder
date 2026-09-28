"""Unit test: fp16 trainable adapters + GradScaler -> reproduced, then fixed.

Exercises the *patched* repo helper minigpt4.models.base_model.BaseModel.force_trainable_fp32
(not a copy of it), using a tiny fp16 GPT2 + peft LoRA as a stand-in for Phi-2.
Run:  cd /data/workspace/MiniGPT-3D && /opt/miniconda3/envs/minigpt5090/bin/python /tmp/verify_guard_unit.py
"""
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import torch
import peft
from transformers import GPT2Config, GPT2LMHeadModel
from peft import LoraConfig, get_peft_model
from minigpt4.models.base_model import BaseModel

print(f"torch {torch.__version__} | peft {peft.__version__} | transformers "
      f"{__import__('transformers').__version__} | gpu {torch.cuda.get_device_name(0)}")

torch.manual_seed(0)
cfg = GPT2Config(vocab_size=256, n_positions=64, n_embd=128, n_layer=2, n_head=4)
base = GPT2LMHeadModel(cfg).half().cuda()
lm = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"],
                                     task_type="CAUSAL_LM"))


def report(tag):
    dts = [(n, str(p.dtype)) for n, p in lm.named_parameters() if p.requires_grad]
    fp16 = [n for n, d in dts if d == "torch.float16"]
    print(f"[{tag}] trainable={len(dts)} fp16={len(fp16)} sample={fp16[:3] or dts[:2]}")
    return fp16


report("after get_peft_model")

x = torch.randint(0, 256, (2, 16)).cuda()


def one_step():
    opt = torch.optim.AdamW([p for p in lm.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    with torch.cuda.amp.autocast(enabled=True):
        loss = lm(x, labels=x).loss
    scaler.scale(loss).backward()
    scaler.step(opt)          # <-- raises "Attempting to unscale FP16 gradients" for fp16 params
    scaler.update()
    opt.zero_grad(set_to_none=True)
    return float(loss)


broke = False
try:
    print("WITHOUT GUARD: scaler.step OK (unexpected), loss =", one_step())
except (RuntimeError, ValueError) as e:
    print("WITHOUT GUARD %s:" % type(e).__name__, str(e).splitlines()[0])
    broke = True

# leftover scaled grads from the failed step must go, otherwise grads stay fp16-typed
lm.zero_grad(set_to_none=True)

converted = BaseModel.force_trainable_fp32(lm, tag="unit-test")
report("after force_trainable_fp32")

ok = False
try:
    print("WITH GUARD: scaler.step OK, loss =", one_step())
    ok = True
except (RuntimeError, ValueError) as e:
    print("WITH GUARD %s:" % type(e).__name__, str(e).splitlines()[0])

passed = broke and ok and len(converted) > 0
print("RESULT:", "PASS (reproduced error + fix works, %d params cast)" % len(converted)
      if passed else "FAIL")
sys.exit(0 if passed else 1)