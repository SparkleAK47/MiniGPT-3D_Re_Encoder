# Training consistency & evaluation parity — 3090 reference chain vs current 5090 working tree

Date: 2026-09-27 · status: **analysis only — no config/asset change applied**
(companion to `docs/amp_fp16_gradscaler_fix.md`, which *is* applied)

## 1. Environment matrix

| | `minigpt_3d` (3090, reference) | `minigpt5090` (this machine) |
|---|---|---|
| python | 3.9 | 3.10 |
| torch | 2.0.0 (+cu117) | 2.7.1+cu128 |
| transformers | 4.36.2 | **4.36.2 (identical)** |
| peft | **0.6.0** | **0.7.1** ← cause of the AMP crash *and* of the dropped Q-Former weights |
| accelerate | 0.21.0 | 0.25.0 |

`env5090.sh` is aspirational, not descriptive: it pins `transformers==4.51.0` (L53) before
overriding it with `4.36.2` (L235) and `peft==0.15.0` (L56), while **0.7.1** is installed.

Measured effects of the peft bump: (i) adapters fp32 → fp16 (the AMP crash, fixed by the fp32
cast of the trainable params); (ii) every LoRA-wrapped parameter renames to
`…base_layer.{weight,bias}`, which silently dropped the BLIP-2 Q-Former initialization of
stages 2–4 (fixed by `remap_peft_base_layer_keys` in `base_model.py` — see
`docs/qformer_pretrained_weights_peft_base_layer.md`). Everything below is drift those two
patches do **not** touch.

## 2. Config drift: reference chain vs working tree

Stored configs of the reference checkpoints vs the current YAMLs
(`docs/validation/ckpt_cfg_diff.py`):

| key | reference checkpoints | working tree |
|---|---|---|
| `model.freeze_pc` | `True` | `False` |
| `model.pc_encoder_ckpt` | `./params_weight/pc_encoder/point_model_V1+random-cls.pth` | `./params_weight/pc_encoder/point_model.pth` |
| `model.train_pc_encoder` | `True` | `True` (unchanged) |

`git diff train_configs/` shows the encoder path was switched in **all four** stage YAMLs
(1 line each) in the working tree; the `freeze_pc` difference is visible in the checkpoints
only (the working-tree YAMLs say `False`). The uncommitted state is anchored by commit
`95feab4` *"Abnormal training"* (2026-09-26).

Consequences (all verified):

1. **Encoder init differs.** `params_weight/pc_encoder/point_model.pth` is stored in
   **float16** (158 fp16 + 2 int64 tensors), whereas `point_model_V1+random-cls.pth` and
   `old/mask-pcpmae.pth` are **float32**. The current YAMLs therefore initialise the
   pc_encoder from an fp16-quantised Point-BERT baseline (the YAML comment calls it
   "原始 Point-BERT (baseline)"), while the reference chain used the fp32 PCP-MAE/V1 encoder.
2. **Different trainable set.** Reference: **500** optimizer params (pc_encoder frozen, but the
   LLM **LN biases** trainable — 96 of them). Current: **558** (pc_encoder unfrozen +154, LN
   biases frozen −96 → net +58). All other module counts are identical
   (`Total Number of trainable parameters: 69277184` in both).
3. **Checkpoints are therefore not interchangeable** — see §3.

## 3. Checkpoint inventory & resume compatibility

(`docs/validation/ckpt_fingerprint.py`, `ckpt_keydiff.py`)

| checkpoint | epoch | model keys | `lora_*` | pc_encoder keys | optimizer params | stored config |
|---|---|---|---|---|---|---|
| `output/pcpmae/stage_1/checkpoint_0.pth` | 0 | 165 | 0 | 160 fp32 | 158 | stage-1 (`only_train_pc_linear`) |
| `output/pcpmae/stage_2/checkpoint_0.pth` | 0 | 667 | 236 **fp32** | **6** | 500 | freeze_pc=True, V1+random-cls, amp=True |
| `output/pcpmae/stage_3/checkpoint_2.pth` | 2 | 667 | 236 **fp32** | 6 | 500 | freeze_pc=True, V1+random-cls, amp=True |
| `output/pcpmae/stage_4/checkpoint_0.pth` | 0 | — | — | — | small (MoE only) | `only_train_MQE` |
| `output/smoke_stage2_ampfix/checkpoint_0.pth` (post-fix) | 0 | 565 | 236 **fp32** | 160 | **558** | current YAML, amp=True |

- LoRA key set **and dtype are identical** between the reference chain and the post-fix run
  (236 fp32) ⇒ the fix restores the reference regime instead of introducing a new one.
- Resume from the post-fix checkpoint works: `resume the checkpoint`, epoch 1 trained,
  `checkpoint_1.pth` written.
- Resume from the **reference** stage-2 checkpoint under the current YAML fails as expected:
  `ValueError: loaded state dict contains a parameter group that doesn't match the size of
  optimizer's group` (`runner_base.py:648`). Model weights load (`strict=False`; the 6
  `pc_encoder.*` keys partially populate the now-trainable encoder), but optimizer groups
  500 vs 558 cannot be mapped. With `freeze_pc: True` + `point_model_V1+random-cls.pth`
  restored the sizes match again.
- The reference `stage_2`/`stage_3` checkpoints carry `amp: True` **and** fp32 LoRA — possible
  only with fp32 adapters, i.e. under peft 0.6.0 (independent confirmation of the AMP cause).

## 4. Evaluation-side state

- `eval_configs/benchmark_evaluation_paper.yaml` points at
  `ckpt: output/pcpmae/stage_3/checkpoint_2.pth` and
  `second_ckpt: output/pcpmae/stage_4/checkpoint_0.pth` (both exist ✔) but at
  `pc_encoder_ckpt: ./params_weight/pc_encoder/mask-pcpmae.pth` — that file **does not exist**;
  it now lives in `params_weight/pc_encoder/old/mask-pcpmae.pth` (the `old/` directory is
  dated 2026-08-03). Since the stage-3/4 checkpoints contain only 6 `pc_encoder.*` keys, the
  eval encoder must come from the YAML ⇒ evaluation needs that path fixed (or the eval YAML
  needs the chosen encoder variant).
- **Stage ≥ 2 evaluations are not comparable across the two environments**: the reference chain
  ran with a BLIP-2-initialised Q-Former, the current stage-2/3/4 runs with a randomly
  initialised one (peft `base_layer` bug, fixed 2026-09-28 — see
  `docs/qformer_pretrained_weights_peft_base_layer.md`). The old-chain metrics are a target for
  the *re-run* chain only.
- Ground truth for the reference chain: `evaluate/V1-random(mask-pcp)_unfreeze/`
  (`evaluation/`, `open-source/`, `traditional/`).

## 5. Recommended Step 3a actions (not applied)

1. Pick one target chain and make it explicit per stage, then commit it:
   either `freeze_pc: True` + `pc_encoder_ckpt: point_model_V1+random-cls.pth` (reference chain)
   or `freeze_pc: False` + the chosen encoder (current state). Do not mix: both resume
   compatibility and metric comparability depend on it.
2. Fix the dead encoder path in the eval YAML (`old/mask-pcpmae.pth`, or the encoder variant
   matching the training stages) and keep eval/train encoders consistent.
3. Re-run the chain from stage 1 with the chosen setting, then evaluate and diff metrics
   against `evaluate/V1-random(mask-pcp)_unfreeze/`.
4. Regardless of the encoder/freeze decision, **stages 2–4 must be re-run**: their Q-Former base
   attention was randomly initialized in the current environment (peft `base_layer` bug) and the
   damaged tensors were never saved → `docs/qformer_pretrained_weights_peft_base_layer.md`.
   Stage 1 is unaffected, so it only needs a re-run if the encoder decision above changes.
5. Cosmetic but worth fixing: `runner_base.train()` raises
   `UnboundLocalError: local variable 'cur_epoch' referenced before assignment` when
   `start_epoch >= max_epoch` (i.e. resuming a checkpoint whose epoch == `max_epoch`
   ⇒ no epochs left to iterate). Bump `max_epoch` when resuming, or guard line 422.

## 6. Caveats / open items

- Bit-identical results across machines are not achievable (kernels, reduction order,
  cu117 vs cu128); metric-level parity is the realistic target.
- The AMP fix changes nothing about *which* parameters are trained, only their dtype, so it
  cannot by itself move metrics — the drift in §2 will.
- The `freeze_pc: False` working-tree chain has not yet been run end-to-end (only a 20-iteration
  smoke test), so its metric parity with the reference chain is still unmeasured. After the
  Q-Former fix that smoke hand-off starts at loss **2.0332** (pre-fix: 7.3235), i.e. it continues
  the stage-1 checkpoint instead of restarting cold.

## 7. Tooling used for the checks above

`docs/validation/ckpt_fingerprint.py` (keys/dtypes/optimizer state),
`docs/validation/verify_qformer_pretrained_load.py` (BLIP-2 Q-Former weights vs the built model,
pre-fix A/B),
`docs/validation/ckpt_cfg_diff.py` (stored-config diff), `docs/validation/ckpt_keydiff.py`
(saved-state key-set diff). All are read-only.