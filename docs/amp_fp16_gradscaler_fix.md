# Stage-2 `ValueError: Attempting to unscale FP16 gradients.` — cause, fix, validation

Date: 2026-09-27 · environment: `minigpt5090` (python 3.10, torch 2.7.1+cu128, RTX 5090 D v2)
· status: **fixed & validated** (patch applied, no config/asset changes).

## 1. Symptom

Stage 2 (`train_configs/MiniGPT_3D/stage_2.yaml`, `amp: True`) dies on the **first optimizer
step**, immediately after the trainable-parameter listing:

```
Traceback (most recent call last):
  File "train.py", line 160, in <module>                        main()
  File "train.py", line 156, in main                            runner.train()
  File "minigpt4/runners/runner_base.py", line 380, in train     train_stats = self.train_epoch(cur_epoch)
  File "minigpt4/runners/runner_base.py", line 444, in train_epoch
  File "minigpt4/tasks/base_task.py", line 116, in train_epoch   return self._train_inner_loop(
  File "minigpt4/tasks/base_task.py", line 232, in _train_inner_loop
      scaler.step(optimizer)
  File ".../torch/amp/grad_scaler.py", line 455, in step         self.unscale_(optimizer)
  File ".../torch/amp/grad_scaler.py", line 342, in unscale_
  File ".../torch/amp/grad_scaler.py", line 264, in _unscale_grads_
      raise ValueError("Attempting to unscale FP16 gradients.")
ValueError: Attempting to unscale FP16 gradients.
```

(Note: the exception class is `ValueError`, raised from `torch/amp/grad_scaler.py:264`.)

## 2. Root cause

**The environment bump `peft 0.6.0 → 0.7.1` — *not* the 5090 code port.**

`BaseModel.init_llm` loads Phi-2 with `torch_dtype=torch.float16` and immediately wraps it
with `get_peft_model(...)`. peft < 0.7 created the injected `lora_A`/`lora_B` matrices in
**fp32**; peft ≥ 0.7 creates them in the **dtype of the wrapped module → fp16**.
`torch.cuda.amp.GradScaler.step()` refuses to unscale fp16 parameter gradients, so the run
dies at step 1 of stage 2.

Stage 1 is unaffected because `only_train_pc_linear: True` freezes the whole LLM (its LoRA
included) — the offending tensors are not trainable there — and the parameters it does train
(`point_2_Qformer_proj` plus the unfrozen `pc_encoder`) are already fp32. That is why the
failure only appears at the **stage-1 → stage-2 transition**.

Arithmetic of the offending set: LLM LoRA = 32 layers × {`query_key_value`, `dense`} × {A,B}
= **128 tensors** ≈ 21 M params. QFormer LoRA (108 tensors) is already fp32 because Bert is
built fp32 → 128 + 108 = **236**, exactly the number of `lora_*` keys stored in the reference
checkpoints.

## 3. Fix (3 hunks, applied)

1. `minigpt4/models/base_model.py` **L138–159** — new `BaseModel.force_trainable_fp32(tag)`:
   casts *only* `requires_grad` fp16 parameters to fp32 and logs
   `[amp] cast N trainable fp16 params to fp32 <tag> | examples: …`.
2. `minigpt4/models/minigpt_v2.py` **L231–234** — call it at the **end of `MiniGPT_3D.__init__`**,
   i.e. after all `requires_grad` decisions and before `from_config` loads the checkpoints.
   Safe by construction: `load_state_dict(..., strict=False)` uses `copy_` semantics, so it
   cannot undo the cast, and `from_config` runs before the optimizer/scaler are created.
3. `train.py` **L98–111** — diagnostics: `dtype=` on each parameter line plus
   `Trainable fp16 params: N (must be 0 for AMP/GradScaler)`.

Cost / risk: only the 128 trainable fp16 tensors change dtype → **≈ +42 MB** GPU memory
(21 M params × 2 B); frozen fp16 LLM weights are untouched. Smoke peak 9.3 GB; production
runs peaked 20.7–24.5 GB ⇒ no OOM risk. Stage 1: cast count is 0 (no-op).
Rollback: `git checkout -- minigpt4/models/base_model.py minigpt4/models/minigpt_v2.py train.py`.

## 4. Validation (all pass)

| # | Check | Result |
|---|-------|--------|
| 1 | fp16 GPT2 + peft 0.7.1 LoRA, one `GradScaler` step | `ValueError: Attempting to unscale FP16 gradients.` (4/4 adapters fp16) |
| 1b | same, after the **repo** helper `force_trainable_fp32()` (`[amp] cast 4 …`) | `scaler.step OK, loss 5.6039` |
| 2 | A/B on the real model + real `stage_2.yaml` (guard off vs on) | `PRE trainable=558 {fp32:430, fp16:128}` → `POST {fp32:558}, fp16=0`; log `[amp] cast 128 trainable fp16 params to fp32 MiniGPT_3D` |
| 3 | End-to-end **pre-fix** (`force_trainable_fp32` monkeypatched off, otherwise identical) | `Trainable fp16 params: 128` → the exact crash above |
| 4 | End-to-end **post-fix** (`amp: True`, 20 iters, scratch dir) | full epoch, loss 7.3235 → 4.8615, `Trainable fp16 params: 0`, `max mem: 9305`, checkpoint written |
| 5 | Resume from the new stage-2 checkpoint (`max_epoch: 2`) | `resume the checkpoint` → epoch 1 trained, `checkpoint_1.pth` written |
| 6 | Checkpoint fingerprint vs the reference chain | 236 `lora_*` keys, all **fp32** (identical key set and dtype), 558 fp32 optimizer states |

Reproduction fidelity: the user's captured stage-2 log (`log_stage_2.txt`) and the smoke run
are fingerprint-identical — `trainable params: 31,457,280 || all params: 2,811,233,280`,
`Total Number of trainable parameters: 69277184`, the same five per-module counts, the same
checkpoint chain (`TinyGPT-V_for_Stage3.pth` → stage-1 `checkpoint_0.pth`) and the same
`[pc_encoder] pc_encoder weights came from full model checkpoint (freeze_pc=False …)`.
In the user's log the last flushed line is the parameter listing, i.e. the crash hit exactly
where the pre-fix reproduction hits it (the traceback itself went to the terminal; stderr is
not captured by `setup_logger()`).

## 5. Ruled out

- **No NaN/inf anywhere** — not in checkpoints, optimizer state or scaler state.
- **Not the ported code**: port commit `47312fb` (*retrain, for 5090*, 2026-07-29) does not
  touch this path; `transformers` is identical in both environments (**4.36.2**); toggling
  *only* the fp32 cast flips the run between crash and success.
- **Corroboration from the reference chain**: `output/pcpmae/stage_2/checkpoint_0.pth` and
  `output/pcpmae/stage_3/checkpoint_2.pth` were trained with `amp: True` yet contain
  **fp32** LoRA (236 keys) — only possible when the adapters themselves were fp32, i.e. under
  peft 0.6.0.

## 6. Re-running the checks

```bash
cd /data/workspace/MiniGPT-3D && export PYTHONPATH=$PWD
PY=/opt/miniconda3/envs/minigpt5090/bin/python
$PY docs/validation/verify_guard_unit.py                                        # 1 / 1b
$PY docs/validation/verify_guard_real.py train_configs/MiniGPT_3D/stage_2.yaml  # 2
$PY train.py --cfg-path docs/validation/stage_2_smoke.yaml                      # 4
$PY docs/validation/train_prefix_repro.py                                       # 3 (expects the crash)
$PY train.py --cfg-path docs/validation/stage_2_resume_new.yaml                 # 5
$PY docs/validation/ckpt_fingerprint.py output/smoke_stage2_ampfix/checkpoint_0.pth  # 6
```

Scratch artifacts created by these runs (safe to delete):
`output/smoke_stage2_ampfix` (796 MB), `output/smoke_resume_new` (796 MB),
`output/smoke_stage2_prefix` (1.5 MB, crash log), `output/smoke_resume_old` (1.5 MB, expected
optimizer-mismatch log) → `rm -rf output/smoke_*`.

## 7. Follow-ups (not part of this patch)

- The *same* peft 0.6.0 → 0.7.1 bump also renamed every LoRA-wrapped parameter to
  `…base_layer.{weight,bias}`, which silently dropped the 108 BLIP-2 Q-Former tensors of
  stages 2–4 → `docs/qformer_pretrained_weights_peft_base_layer.md` (**fixed** 2026-09-28;
  stages 2–4 must be re-run, stage 1 stays valid). Validation:
  `docs/validation/verify_qformer_pretrained_load.py`.
- Config/asset parity for reproducible metrics → `docs/training_consistency_and_eval_parity.md`.
- Resuming the **old** reference stage-2 checkpoint under the current config fails on
  optimizer-state load (500 vs 558 params) — a config-drift consequence, unrelated to this fix.