# Validation kit for the environment-regression fixes (AMP / GradScaler, Q-Former pretrained load)

Run everything **from the repository root** with `PYTHONPATH=$PWD` (the repo has no
`setup.py` entry for these scripts) and the 5090 interpreter:

```bash
cd /data/workspace/MiniGPT-3D && export PYTHONPATH=$PWD
PY=/opt/miniconda3/envs/minigpt5090/bin/python
```

| file | what it does | expected result |
|---|---|---|
| `verify_guard_unit.py` | tiny fp16 GPT2 + peft 0.7.1 LoRA; one `GradScaler` step **without** and **with** the repo helper | `ValueError: Attempting to unscale FP16 gradients.` then `scaler.step OK` + `[amp] cast 4 …`; exits 0 |
| `verify_guard_real.py <cfg.yaml>` | builds the real `MiniGPT_3D` twice (guard disabled vs enabled) and reports trainable dtypes | `PRE … fp16_trainable=128` → `POST … fp16_trainable=0`; prints the `[amp] cast 128 …` line; exits 0 |
| `verify_model_dtypes.py <cfg.yaml>` | builds the model once and asserts no trainable fp16 param | `dtype histogram {'torch.float32': 558}`, `RESULT: PASS` |
| `verify_qformer_pretrained_load.py <cfg.yaml>` | builds the real `MiniGPT_3D` twice (pre-fix = remap dropped vs the repo code) and compares every BLIP-2 Q-Former key with the model weights | stages 2–4: PRE `wrapped deviating=108` (max 1.35e+01) + 1 missing-key warning → POST `wrapped=108, wrapped deviating=0`, no warning (`absmax 0.18265417 / 0.11476848`); stage 1: `wrapped=0`, 0 deviations ⇒ PASS/`exit 0` for `stage_1..4.yaml` |
| `train_prefix_repro.py` | runs `train.py` with `MiniGPT_3D.force_trainable_fp32` monkeypatched to a no-op (pre-fix behaviour) | `Trainable fp16 params: 128` then the original crash (`base_task.py:232`); exits non-zero on purpose |
| `stage_2_smoke.yaml` | 20-iteration stage-2 smoke run, `amp: True`, output `./output/smoke_stage2_ampfix` | full epoch, no crash; first loss ≈**2.0** (continues the stage-1 checkpoint; it was ≈7.3 before the Q-Former fix — see `docs/qformer_pretrained_weights_peft_base_layer.md`) |
| `stage_2_smoke_prefix.yaml` | same but 2 iterations, output `./output/smoke_stage2_prefix` (used by `train_prefix_repro.py`) | crash, as above |
| `stage_2_resume_new.yaml` | resumes `./output/smoke_stage2_ampfix/checkpoint_0.pth` with `max_epoch: 2` | `resume the checkpoint`, epoch 1 trained, `checkpoint_1.pth` |
| `stage_2_resume_old.yaml` | resumes the reference `./output/pcpmae/stage_2/checkpoint_0.pth` | model loads, optimizer state load raises (500 vs 558 params) — config drift, not the fix |
| `ckpt_fingerprint.py <ckpt…>` | structure / dtype / optimizer-state fingerprint of checkpoints | see `docs/training_consistency_and_eval_parity.md` §3 |
| `ckpt_cfg_diff.py <old> <new>` | diff of the configs stored inside two checkpoints | drift table in §2 of the same doc |
| `ckpt_keydiff.py <old> <new>` | diff of the saved `model` state key sets | which params each run tracked |

Notes

- GPU required for everything except the three `ckpt_*.py` tools; each full run builds Phi-2
  and takes ~2–4 min (the two `verify_*_real`/`verify_qformer_*` probes only build the model:
  ≈20 s per build, i.e. ~1 min per invocation).
- The smoke runs write ≈0.8 GB per checkpoint. Clean up with `rm -rf output/smoke_*`.
- Outputs are logged to `/tmp/smoke_stage2.out`, `/tmp/smoke_prefix.out`,
  `/tmp/resume_new2.out`, `/tmp/resume_old2.out` when using the commands in
  `docs/amp_fp16_gradscaler_fix.md` §6.