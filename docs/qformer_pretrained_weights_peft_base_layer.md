# Stage 2 restarts from loss ≈7.5 — BLIP-2 Q-Former weights silently dropped by the peft `base_layer` rename

Date: 2026-09-28 · environment: `minigpt5090` (python 3.10, torch 2.7.1+cu128, RTX 5090 D v2)
· status: **fixed & validated** (patch applied, no config/asset change). Stage 1 stays valid as
is; **stages 2–4 must be re-run** (their Q-Former base weights are unrecoverable).

Companion docs: `docs/amp_fp16_gradscaler_fix.md` (same peft 0.6.0 → 0.7.1 bump, different
symptom), `docs/training_consistency_and_eval_parity.md` (environment/config drift).

## 1. Symptom

Stage 2 trains from a loss that is *worse* than stage 1 ever saw, i.e. the stage-1 weights
behave as if they had not been learned:

| run (same chain: `TinyGPT-V_for_Stage3.pth` → stage-1 `checkpoint_0.pth`) | iter 0 | iteration 19 (20-iter smoke) |
|---|---|---|
| stage 1 (`QFormer_lora_r: 0`) | 7.2286 | ≈1.8 at the end of its epoch |
| stage 2, old environment `minigpt_3d` (peft 0.6.0) | **2.2754** | continuous |
| stage 2, `minigpt5090` (peft 0.7.1) **pre-fix** | **7.5342** (7.3235 in the smoke) | 4.8615 |
| stage 2, `minigpt5090` **post-fix** (`docs/validation/stage_2_smoke.yaml`) | **2.0332** | 2.2850 |

The port itself is innocent: `git diff f024168 47312fb -- minigpt4/models/minigpt_v2.py` is
**empty**, and the same code in the old environment continues at 2.28 while the new environment
restarts at 7.53. The only difference is the interpreter/package set.

Peak memory is unchanged pre/post fix (8631 → 9305 MB in the smoke, identical to the AMP-fix
smoke), so this is not a dtype/budget regression but *initialization*.

## 2. Root cause

`BaseModel.load_from_pretrained` loads BLIP-2 with `load_state_dict(..., strict=False)` and the
missing-key report was commented out (`base_model.py`, pre-patch):

```python
state_dict = checkpoint["model"]
msg = self.load_state_dict(state_dict, strict=False)
# logging.info("Missing keys {}".format(msg.missing_keys))   <-- silent
```

The Q-Former is built **and LoRA-injected** before that load, inside
`MiniGPT_3D.init_Qformer` (`inject_adapter_in_model`), and the BLIP-2 load follows at
`minigpt_v2.py:102`. The key names therefore have to match the *wrapped* model:

| peft | wrapper class | parameter name of a target module |
|---|---|---|
| ≤ 0.6.0 (old env) | `class Linear(nn.Linear, LoraLayer)` | `…self.query.weight` ✔ plain |
| ≥ 0.7.1 (new env) | `class Linear(nn.Module, LoraLayer)` with `self.base_layer` | `…self.query.base_layer.weight` ✘ renamed |

With `QFormer_lora_r: 8` and `QFormer_lora_module: ["query", "key", "value"]` peft wraps
**108** Q-Former tensors (the same 108 modules whose adapters make up the "QFormer LoRA = 108
tensors" of the AMP doc). In the new environment every one of the 108 plain BLIP-2 keys failed
to match, was silently dropped and the Q-Former self-/cross-attention kept the random
initialization produced by `from_config`. `attention.output.dense` is not a LoRA target and
stayed correct — which is exactly the fingerprint below.

Why this shows up only at the stage-1 → stage-2 transition:

- stage 1 uses `QFormer_lora_r: 0` ⇒ **no** wrapping ⇒ BLIP-2 loads into the plain Q-Former,
  and stage 1 starts from 7.2 for a different reason (untrained projectors).
- stage 2/3/4 use `r: 8` ⇒ the 108 tensors are wrapped ⇒ they are dropped *and* the Q-Former is
  frozen for those stages, so training cannot repair them either.
- The Q-Former base weights are never written into a stage checkpoint (they are frozen during
  stages 2–4), so the corruption is invisible in **every** artefact: only the *seed* decides
  which random Q-Former the new run got, and the eval/train scripts stay mutually consistent
  only because they use the same seed. Any RNG-stream change silently drifts them apart.

The chain is therefore: peft 0.7.1 → wrapped names → 108 BLIP-2 keys dropped → random Q-Former
attention → the point-cloud pathway behaves like a fresh model while the rest of stage 1 is
intact → loss jumps back to ≈7.5.

## 3. Fix (方案 B: one generic key remap + one targeted warning)

1. `minigpt4/models/base_model.py` — new module-level helper
   `remap_peft_base_layer_keys(state_dict, model_state_dict)`: for every checkpoint key that the
   model does not have, it retries with `.base_layer` inserted before the trailing
   `.weight`/`.bias`, and keeps the key unchanged when that variant does not exist. It is a
   **no-op for un-wrapped modules and for peft ≤ 0.6**, logs
   `[load] remapped N LoRA-wrapped key(s) onto peft base_layer, e.g. …` when it fires.
2. `minigpt4/models/base_model.py` — `load_from_pretrained(url_or_filename, key_remap=None)`
   applies the callback to `checkpoint["model"]` before `load_state_dict(..., strict=False)`,
   so the fix lives in one place and any caller can opt in (default `None` ⇒ old behaviour).
3. `minigpt4/models/base_model.py` — new `warn_missing_qformer_keys(msg, source)`: reports the
   missing `Qformer.*` keys that are not peft adapters (`*.lora_*`), i.e. exactly the frozen
   base weights that have no other source than the BLIP-2 checkpoint. It replaces the
   commented-out missing-key log for this class of keys.
4. `minigpt4/models/minigpt_v2.py` — passes `key_remap=remap_peft_base_layer_keys` at the three
   load sites: the BLIP-2 load (`:102`) and the `ckpt`/`second_ckpt` loads in `from_config`
   (legacy full-model checkpoints may use the un-wrapped names too).

The change consumes no randomness, creates no module and touches no dtype ⇒ the stage-1 build
(and its already-trained checkpoint) is bit-identical pre/post fix, and stage-2/3/4 keep the
exact same parameter set (`model keys=1145`, `trainable params: 31,457,280 || all params:
2,811,233,280` unchanged).

Why a remap instead of re-ordering the code: moving the LoRA injection after the BLIP-2 load
(the alternative "方案 A") additionally changes when `inject_adapter_in_model` consumes the RNG
stream, which would shift the initialization of *every* trainable tensor — that is the
mechanism behind the silent eval/train drift described in §5, so it is deliberately avoided.

## 4. Validation

### 4.1 Fingerprints — `docs/validation/verify_qformer_pretrained_load.py`

Builds the real model twice (Run A = pre-fix behaviour by dropping the `key_remap` argument,
Run B = repository code), both with `seed: 42`, and compares all 255 BLIP-2 Q-Former keys with the
model's tensors. Asserted are the **peft-wrapped** tensors: they are frozen for stages ≥ 2 and
never stored in a checkpoint, so they must be exactly the BLIP-2 values. The other Q-Former keys
are LayerNorms (`train_QFormer_norm: True`) that the earlier stage trained and saved — a deviation
there is expected and is only reported.

| frozen `Qformer.bert.encoder.layer.0` fingerprint \ metric | BLIP-2 | stage 1 (`r: 0`) | stages 2–4 PRE-fix (`r: 8`) | stages 2–4 POST-fix |
|---|---|---|---|---|
| `attention.self.query.weight` (absmax) | 0.18265 | **0.18265417** | 0.09625707 | **0.18265417** |
| `crossattention.self.query.weight` (absmax) | 0.11477 | **0.11476848** | 0.10058621 | **0.11476848** |
| `attention.output.dense.weight` (not a LoRA target) | 0.41106 | 0.41105857 | 0.41105857 | 0.41105857 |
| Q-Former keys wrapped by peft | – | 0 | 108 | 108 |
| **wrapped keys deviating from BLIP-2** | – | 0 | **108** (max 1.352e+01) | **0** |
| other Q-Former keys deviating (inherited trained LNs) | – | 0 | 0 (stage 2) / 30 (stages 3–4) | 0 (stage 2) / 30 (stages 3–4) |
| `[load] … Q-Former key(s) MISSING` warning | – | 0 | **1** (108 keys) | 0 |
| `model keys` | – | 1037 | 1145 (1157 at stage 4) | same |

All four configs (`stage_1..4.yaml`) end with `RESULT: PASS` / `exit 0`, and Run A still
reproduces the damage (0.09625707 / 0.10058621 are exactly the random values measured before the
fix ⇒ the mechanism is understood, not patched by luck). The stage-1 run reports `peft-wrapped
Q-Former keys=0` and `PRE-FIX wrapped deviating=0 | POST-FIX wrapped deviating=0`, i.e. **stage 1
was never affected and its already-trained checkpoint remains valid**; stages 3–4 additionally
show the 30 inherited LayerNorm deltas, which is precisely the set of Q-Former keys their
checkpoints carry (`ckpt_keydiff.py`).

### 4.2 End-to-end hand-off (`docs/validation/stage_2_smoke.yaml`)

Identical config, identical chain (`TinyGPT-V_for_Stage3.pth` → `output/pcpmae/stage_1/
checkpoint_0.pth`), `amp: True`, 20 iterations, only the code differs:

| | iter 0 | iter 19 | max mem | checkpoints |
|---|---|---|---|---|
| pre-fix (`/tmp/smoke_stage2.out`) | 7.3235 | 4.8615 | 8631 → 9305 MB | `output/smoke_stage2_ampfix` |
| post-fix (`/tmp/smoke_qformer_fix.out`) | **2.0332** | 2.2850 | 8631 → 9305 MB | `output/smoke_qformer_fix` |

The post-fix run starts where stage 1 left off (≈2) instead of restarting at ≈7.5, with an
unchanged memory profile and the same trainable set (`Trainable fp16 params: 0`).

### 4.3 Checkpoint invariance

`docs/validation/ckpt_keydiff.py output/smoke_stage2_ampfix/checkpoint_0.pth
output/smoke_qformer_fix/checkpoint_0.pth` → 565 vs 565 entries, **0 keys only in OLD, 0 only in
NEW** (236 `lora_*`, 31 `Qformer.bert.*`, 160 `pc_encoder.*`, 129 `llama_model.base_model.*`…);
`ckpt_fingerprint.py` on the post-fix checkpoint: 236 fp32 LoRA keys (AMP fix intact), 558 fp32
optimizer states, `scaler` present. The fix changes values, not the parameter set — and
confirms §5: the Q-Former base weights are in **neither** checkpoint.

## 5. Why nothing caught this earlier (and what it means for comparability)

- The Q-Former base attention is frozen during stages 2–4 and therefore **never saved**; its
  only source is `blip2_pretrained_flant5xxl.pth`. A wrong initialization is thus invisible in
  checkpoints, in the optimizer state and in the trainable-parameter listing (the count is
  identical: 31,457,280 / 558 params).
- The masked missing-key log (`base_model.py`) swallowed the 108 dropped keys until this patch.
  The new warning fires even before `setup_logger()` (it is emitted through the root logger's
  last-resort handler — verified: the `WARNING` line appears on stderr during `__init__`, while
  plain `INFO` lines are dropped), so the failure mode can no longer be silent.
- Train and eval builds share the seed, so they *mutually* agreed on the same wrong Q-Former —
  which is why the corruption looked like a legitimate "hard start" rather than a bug. Any
  change to the RNG stream (new flag, different batch order, torch version) would have silently
  desynchronised train and eval.
- Consequence: old-environment stage-2/3/4 checkpoints and new-environment ones were trained
  with **different Q-Former networks**. Loss/eval comparisons across the two eras for stages
  ≥ 2 are not meaningful — a plausible confounder of earlier eval-parity discrepancies
  (`docs/training_consistency_and_eval_parity.md` §4).

## 6. Consequences & next steps

1. **Re-run stages 2 → 4.** The corrupted Q-Former attention was never saved, so nothing can be
   recovered from the existing checkpoints; start from the stage-1 checkpoint again
   (`bash run_all.sh` or the four `train.py --cfg-path train_configs/MiniGPT_3D/stage_N.yaml`).
   The existing stage-1 checkpoint stays valid (§4.1).
2. Keep the guard in the loop: run `docs/validation/verify_qformer_pretrained_load.py` after any
   environment/dependency change (peft, torch, transformers) and after touching
   `init_Qformer`/`load_from_pretrained`.
3. Do not compare stage ≥ 2 metrics against old-environment runs (different Q-Former network,
   see §5); compare against runs of the fixed chain only.
4. Optional, not part of this patch: the TinyGPT checkpoint stores its LLM-LoRA keys as
   `…lora_A.weight` while the peft-wrapped model expects `…lora_A.default.weight` ⇒ they are
   silently skipped as well. Harmless today (`lora_B` is zero and the LLM is frozen in every
   stage), but it can be repaired with `peft.utils.set_peft_model_state_dict` or by remapping
   `lora_(A|B).` → `lora_\1.default.`.

## 7. Re-running the checks

```bash
cd /data/workspace/MiniGPT-3D && export PYTHONPATH=$PWD
PY=/opt/miniconda3/envs/minigpt5090/bin/python

# 1) fingerprints, pre-fix vs post-fix A/B (GPU, ~20 s per build, exits non-zero on regression)
for s in 1 2 3 4; do
    $PY -u docs/validation/verify_qformer_pretrained_load.py train_configs/MiniGPT_3D/stage_$s.yaml
done

# 2) end-to-end hand-off from the stage-1 checkpoint (first loss must be ≈2, not ≈7.5)
$PY train.py --cfg-path docs/validation/stage_2_smoke.yaml \
    --options run.output_dir=./output/smoke_qformer_fix run.job_name=minigpt3d_smoke_qformer_fix

# 3) checkpoint invariance (key sets only; read-only)
$PY docs/validation/ckpt_keydiff.py output/smoke_stage2_ampfix/checkpoint_0.pth \
                                  output/smoke_qformer_fix/checkpoint_0.pth
$PY docs/validation/ckpt_fingerprint.py output/smoke_qformer_fix/checkpoint_0.pth
```

Scratch artifacts (`output/smoke_qformer_fix`, 796 MB) are disposable: `rm -rf output/smoke_*`.