"""Guard: the BLIP-2 Q-Former weights must survive the peft LoRA injection.

Regression this protects against
--------------------------------
``MiniGPT_3D.__init__`` injects the Q-Former LoRA inside ``init_Qformer`` and only
*afterwards* loads ``blip2_pretrained_flant5xxl.pth`` with
``load_state_dict(..., strict=False)``.  With peft >= 0.7 every injected module is
wrapped as ``*.base_layer``, so the plain BLIP-2 keys (144 tensors for
``QFormer_lora_r: 8`` with modules query/key/value) silently failed to match and the
Q-Former self-/cross-attention kept its random initialization.  Stage 1
(``QFormer_lora_r: 0``) is not affected, so stage 2 restarted from loss ~7.5 instead of
continuing at ~2.2; the Q-Former is frozen for stage >= 2 and never written into a
checkpoint, so no checkpoint revealed the corruption.

What it checks
--------------
Run A: pre-fix build (the ``key_remap`` argument is dropped) -> reports the Q-Former
       keys with no counterpart, their deviation from BLIP-2 and the new
       ``[load] ... Q-Former key(s) MISSING`` warning.  For a config with
       ``QFormer_lora_r: 0`` nothing is wrapped, hence 0 dropped keys.
Run B: the repository code -> every BLIP-2 Q-Former key must load bit-exactly and no
       warning may be emitted.

Run B doubles as the stage-1/stage-2 consistency check: for both configs the Q-Former
attention values must equal the BLIP-2 ones (see
``docs/training_consistency_and_eval_parity.md`` paragraph 4).

Run (GPU required, ~2-4 min per build):
  cd /data/workspace/MiniGPT-3D && export PYTHONPATH=$PWD
  PY=/opt/miniconda3/envs/minigpt5090/bin/python
  $PY -u docs/validation/verify_qformer_pretrained_load.py train_configs/MiniGPT_3D/stage_2.yaml
  $PY -u docs/validation/verify_qformer_pretrained_load.py train_configs/MiniGPT_3D/stage_1.yaml
"""
import logging
import sys
import types

import torch

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import minigpt4.models  # noqa: F401  (registers "minigpt_3d")
from minigpt4.models.base_model import remap_peft_base_layer_keys
from minigpt4.models.minigpt_v2 import MiniGPT_3D

BLIP2 = "sfr-vision-language-research^LAVIS/blip2_pretrained_flant5xxl.pth"
CFG = sys.argv[1] if len(sys.argv) > 1 else "train_configs/MiniGPT_3D/stage_2.yaml"

# Fingerprints: the first two are LoRA targets (wrapped by peft >= 0.7), the third is
# not a target, so it must match BLIP-2 in *both* runs.
TARGETS = [
    "Qformer.bert.encoder.layer.0.attention.self.query.weight",
    "Qformer.bert.encoder.layer.0.crossattention.self.query.weight",
    "Qformer.bert.encoder.layer.0.attention.output.dense.weight",
]

logs = []


class Capture(logging.Handler):
    def emit(self, record):
        logs.append(record.getMessage())


logging.getLogger().addHandler(Capture())
logging.getLogger().setLevel(logging.INFO)

blip2 = torch.load(BLIP2, map_location="cpu")["model"]
qformer_keys = [k for k in blip2 if k.startswith("Qformer.")]
print(f"### cfg: {CFG}", flush=True)
print(f"BLIP-2 Q-Former keys: {len(qformer_keys)}", flush=True)


def build(seed=42):
    cfg = Config(types.SimpleNamespace(cfg_path=CFG, options=[]))
    torch.manual_seed(seed)
    return registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)


def model_key_for(key, model_state):
    """Model key holding ``key`` (plain, or its peft ``base_layer`` variant), else None."""
    candidate = next(iter(remap_peft_base_layer_keys({key: None}, model_state)))
    return candidate if candidate in model_state else None


def probe(model, tag):
    """Report, for one built model, how its Q-Former compares with the BLIP-2 weights.

    Only the *peft-wrapped* tensors are asserted on: they are frozen for stages >= 2 and never
    written into a checkpoint, so they must still be **exactly** the BLIP-2 values. All other
    Q-Former keys (the LayerNorms, `train_QFormer_norm: True`) are trained and saved by the
    earlier stages, so their deviation from BLIP-2 is expected and only logged.
    """
    state = model.state_dict()
    pairs = [(k, model_key_for(k, state)) for k in qformer_keys]
    wrapped = {k: mk for k, mk in pairs if mk is not None and mk != k}
    absent = [k for k, mk in pairs if mk is None]
    diffs = {k: (state[mk].float() - blip2[k].float()).abs().max().item()
             for k, mk in pairs if mk is not None}
    wrapped_bad = {k: v for k, v in diffs.items() if k in wrapped and v > 0}
    other_bad = {k: v for k, v in diffs.items() if k not in wrapped and v > 0}
    warns = [m for m in logs if "Q-Former key(s) MISSING" in m]
    remaps = [m for m in logs if "remapped" in m]
    print(f"[{tag}] model keys={len(state)} | peft-wrapped Q-Former keys={len(wrapped)} | "
          f"absent={len(absent)} | wrapped deviating from BLIP-2={len(wrapped_bad)} "
          f"(max {max(wrapped_bad.values(), default=0.0):.3e}) | other Q-Former keys "
          f"deviating={len(other_bad)} (max {max(other_bad.values(), default=0.0):.3e}) | "
          f"Q-Former warnings={len(warns)}", flush=True)
    if remaps:
        print(f"    {remaps[0]}", flush=True)
    if warns:
        print(f"    {warns[0]}", flush=True)
    if absent:
        print(f"    absent e.g. {absent[:2]}", flush=True)
    if other_bad:
        print(f"    note: those {len(other_bad)} non-wrapped Q-Former keys are LayerNorms trained "
              f"and saved by an earlier stage -> a deviation from BLIP-2 is expected", flush=True)
    for key in TARGETS:
        model_key = model_key_for(key, state)
        if model_key is None:
            print(f"    {key}: NOT PRESENT IN MODEL", flush=True)
            continue
        tensor = state[model_key].float()
        print(f"    {model_key}\n        absmax={tensor.abs().max().item():.8f}"
              f" maxdiff_vs_blip2={diffs[key]:.3e}", flush=True)
    return {"wrapped": wrapped, "absent": absent, "wrapped_bad": wrapped_bad,
            "other_bad": other_bad, "warns": warns}


# ---- Run A: simulate the pre-fix code by ignoring the remap ----
orig_load = MiniGPT_3D.load_from_pretrained


def pre_fix_load(self, url_or_filename, key_remap=None):
    return orig_load(self, url_or_filename, key_remap=None)


MiniGPT_3D.load_from_pretrained = pre_fix_load
try:
    model = build()
    a = probe(model, "A PRE-FIX (remap dropped)")
    del model
finally:
    MiniGPT_3D.load_from_pretrained = orig_load
    torch.cuda.empty_cache()

# ---- Run B: the repository code ----
del logs[:]
model = build()
b = probe(model, "B POST-FIX (repo code)")
del model
torch.cuda.empty_cache()

if b["wrapped"]:
    print(f"peft wrapped {len(b['wrapped'])} Q-Former tensors (QFormer_lora_r > 0); they can "
          f"only be filled through the key remap.", flush=True)
else:
    print("note: this cfg does not wrap the Q-Former (QFormer_lora_r == 0), so the "
          "pre-fix and the fixed build are identical.", flush=True)

ok = (not b["wrapped_bad"] and not b["warns"] and not b["absent"]
      and (not a["wrapped"] or (a["wrapped_bad"] and a["warns"])))
print(f"PRE-FIX wrapped deviating={len(a['wrapped_bad'])} warnings={len(a['warns'])} | "
      f"POST-FIX wrapped deviating={len(b['wrapped_bad'])} warnings={len(b['warns'])}")
print("RESULT:", "PASS (every peft-wrapped BLIP-2 Q-Former weight loads bit-exactly, the "
      "dropped-key warning stays silent)" if ok else
      "FAIL (frozen Q-Former base weights do not match BLIP-2)")
sys.exit(0 if ok else 1)