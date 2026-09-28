"""End-to-end PRE-FIX reproduction: run train.py with force_trainable_fp32 disabled.

Everything else is the patched code, so the only difference vs the passing smoke run
is the absence of the fp32 cast -> expects the original
'Attempting to unscale FP16 gradients.' crash at the first optimizer step.
"""
import os
import runpy
import sys

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stage_2_smoke_prefix.yaml")
sys.argv = ["train.py", "--cfg-path", CFG]

import minigpt4.models.minigpt_v2 as mv

print("### disabling MiniGPT_3D.force_trainable_fp32 (simulating the pre-patch code)",
      flush=True)
mv.MiniGPT_3D.force_trainable_fp32 = lambda self, tag="": []

runpy.run_path("/data/workspace/MiniGPT-3D/train.py", run_name="__main__")