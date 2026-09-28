"""Build the *real* MiniGPT_3D model from a repo config and verify the AMP invariant:
no trainable parameter may be fp16 (GradScaler cannot unscale fp16 grads).

Run:  cd /data/workspace/MiniGPT-3D && /opt/miniconda3/envs/minigpt5090/bin/python \
        /tmp/verify_model_dtypes.py train_configs/MiniGPT_3D/stage_2.yaml
"""
import collections
import logging
import sys
import types

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

import torch
from minigpt4.common.config import Config
from minigpt4.common.registry import registry
import minigpt4.models  # noqa: F401  registers MiniGPT_3D

CFG = sys.argv[1] if len(sys.argv) > 1 else "/data/workspace/MiniGPT-3D/train_configs/MiniGPT_3D/stage_2.yaml"
# same contract as train.py: Config(args) with args.cfg_path / args.options
cfg = Config(types.SimpleNamespace(cfg_path=CFG, options=[]))

model_cls = registry.get_model_class(cfg.model_cfg.arch)
print(f"config: {CFG}")
print(f"model class: {model_cls.__name__} | arch: {cfg.model_cfg.arch} | "
      f"amp: {cfg.run_cfg.get('amp', None)} | freeze_pc: {cfg.model_cfg.get('freeze_pc', None)} | "
      f"only_train_pc_linear: {cfg.model_cfg.get('only_train_pc_linear', None)} | "
      f"only_train_MQE: {cfg.model_cfg.get('only_train_MQE', None)}")

model = model_cls.from_config(cfg.model_cfg)
model.eval()

trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
hist = collections.Counter(str(p.dtype) for _, p in trainable)
fp16 = [n for n, p in trainable if p.dtype == torch.float16]
by_module = collections.Counter(n.split(".")[0] for n, _ in trainable)
n_lora = sum(1 for n, _ in trainable if "lora_" in n)

print("---- trainable summary ----")
print(f"trainable tensors : {len(trainable)}")
print(f"dtype histogram   : {dict(hist)}")
print(f"lora_" + " tensors       :", n_lora)
print(f"by top-level module: {dict(by_module)}")
print(f"fp16 trainable     : {len(fp16)} {fp16[:5]}")

passed = len(fp16) == 0
print("RESULT:", "PASS (0 fp16 trainable -> scaler.step safe)" if passed
      else f"FAIL ({len(fp16)} fp16 trainable)")
sys.exit(0 if passed else 1)