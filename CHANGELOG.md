# Changelog

- 2026-07-24 — plan written, repo scaffolding started.
- 2026-08-01 — EDA run on real data (12120 train / 5195 test rows). Findings:
  class balance near-even (ratio 1.08, no resampling needed); English 56.7%
  of train, other 14 languages ~340-410 rows each (bg/tr/de smallest — watch
  these for weak per-language accuracy); no train/test language mismatch; 0
  exact-duplicate (premise, hypothesis) pairs; combined premise+hypothesis
  p95 token length (xlm-roberta-large tokenizer) = 74 → `max_length: 128` set
  in `configs/base.yaml`. Full report: `experiments/eda_report.md`.
- 2026-08-02 — pipeline code written: `src/{utils,data,eda,profiling,train,infer}.py`,
  `configs/{base,smoke}.yaml`, `experiments/runs.md` initialized. Smoke test
  left for user to run locally (instructions in Readme.md) rather than run
  from this session.
- 2026-08-08 — EDA report path moved to `experiments/eda_report.md` (versioned),
  plots/csv stay `outputs/eda/` (gitignored).
- 2026-08-10 — optional ClearML logging added (`src/clearml_logger.py`,
  `clearml.enabled` flag in config, off by default, separate optional dep).
- 2026-08-11 — smoke test run by user on Mac (mps, `xlm-roberta-base` for
  speed/RAM — `smoke.yaml` deliberately uses a smaller model than `base.yaml`'s
  `xlm-roberta-large`). Pipeline verified end to end. Missing deps found and
  pinned in `pyproject.toml`: `accelerate` (HF Trainer hard requirement, was
  missing entirely), `setuptools==75.6.0` pinned (v81+ dropped `pkg_resources`,
  which `tensorboard`'s CLI needs at runtime), `torch-tb-profiler` (needed for
  tensorboard's PYTORCH_PROFILER tab to read `torch.profiler` trace.json —
  without it tensorboard shows "no dashboards active").
- 2026-08-19 — read profiler trace.json manually (tensorboard UI trace-tree
  parser threw non-fatal errors on the mps trace): `Optimizer.step#AdamW.step`
  was ~94% of profiled step time, forward/backward (`scaled_dot_product_attention`)
  negligible by comparison — AdamW likely running non-fused/non-foreach on mps,
  one kernel launch per parameter. Mac-only artifact, not worth fixing — 5090
  (cuda) run should default to fused AdamW. Re-profile on real `base.yaml` run
  on the GPU server, not off the smoke test, before drawing conclusions there.
- 2026-08-24 — all filesystem paths moved out of code into config `paths:`
  block (`data_dir`, `output_root`, `experiments_dir`) — `src/data.py`,
  `src/eda.py`, `src/train.py`, `src/infer.py` no longer hardcode any path.
  `src/eda.py` now takes `--config` (default `configs/base.yaml`) instead of
  running argument-free. Portability across machines is a config edit only.
- 2026-09-07 — first real profiling runs on the 5090 (`base.yaml`, cuda, bf16).
  Baseline step time ~144 ms, GPU util 89%, "Other" 8.3% of step. Trace showed
  a per-step device→host sync storm (`aten::is_nonzero` ~17 ms each,
  `aten::_local_scalar_dense` ~790 calls/step) plus fixed-`max_length` padding
  wasting ~40% of every gemm (EDA p95 = 74 tokens vs `max_length` 128).
- 2026-09-07 — training input pipeline reworked for dynamic padding:
  `src/data.py` `NLIDataset.__getitem__` no longer pads or returns tensors
  (`padding="max_length"` + `return_tensors="pt"` removed) — emits variable-length
  encodings, labels as plain ints. `src/train.py` adds
  `data_collator=DataCollatorWithPadding(tokenizer)` (pads to batch max),
  `group_by_length=True` (batch similar lengths, shrink batch max),
  `dataloader_num_workers` (new `training.num_workers` config key, 8 on server),
  and `logging_nan_inf_filter=False` (kills the per-step isnan/isinf loss sync).
  `configs/{base,smoke}.yaml` gain `num_workers`.
- 2026-09-07 — `src/infer.py` given the same `DataCollatorWithPadding` collate_fn
  on its `DataLoader` — without it the ragged (now unpadded) `NLIDataset` batches
  crash `default_collate` ("each element in list of batch should be of equal size").
- 2026-09-08 — re-profiled after the pipeline rework: step time 144 → 83 ms
  (−42%), GPU util 89 → 94%, SM efficiency 86 → 91%, "Other" 8.3 → 1.8%.
  Pipeline is now GPU-compute-bound (kernel ~93% of step), DataLoader 0%.
  `tokenizers` fork warning is cosmetic (fast tokenizer used in parent before
  DataLoader fork) — silence with `TOKENIZERS_PARALLELISM=false` if wanted.
- 2026-09-08 — batch size experiment: `batch_size` 64 → 160. Step time 83 → 135 ms
  but throughput +55% (0.77 → 1.19 samples/ms); occupancy stayed low (~28%,
  register-bound large-tile gemm, not a real stall). Kept as a candidate, not
  locked in — larger effective batch needs LR re-tuning.
- 2026-09-08 — first end-to-end training runs on the 5090, logged in
  `experiments/runs.md`:
  - run with 5 epochs / batch 160 / warmup_ratio 0.06: **collapsed to majority
    class** (val_acc 0.34, eval_loss flat at ln(3)). Cause: warmup 6% of a short
    schedule finished by ~step 4, near-full LR hit at step 50 with grad_norm 36 —
    classic XLM-R-large fine-tune instability.
  - run with 35 epochs / batch 160 (warmup 6% = ~145 steps, gentle start): trained
    fine, best val_acc **0.824** @ epoch 15, but heavy overfit after ~epoch 8
    (train loss → 0.0007, eval_loss 0.54 → 1.58, eval_acc plateau 0.80–0.82).
    First working submission scored **0.80 on the test leaderboard**.
- 2026-09-08 — takeaways for the next run (not yet applied): cut epochs to ~10,
  add `EarlyStoppingCallback(patience=3)`, `warmup_ratio: 0.1`, try `lr: 1e-5`
  for large-model stability, disable profiling. Candidate models to try beyond
  `xlm-roberta-large`: `MoritzLaurer/mDeBERTa-v3-base-xnli-*` and
  `joeddav/xlm-roberta-large-xnli` (pre-tuned on XNLI, same 15 languages).
  Cross-validation (stratified K-fold on label×lang) worth adding once the
  recipe is stable, not before.
- 2026-09-16 — `ensembles/` added: `blend_answers.py` majority-votes across
  multiple submission csvs (ties broken by the best single model's vote).
  `src/infer_probs.py` added alongside `infer.py` — same checkpoint/config
  path but writes per-class softmax probabilities (`id,prob_0,prob_1,prob_2`,
  6 decimals) instead of argmax labels, for ensembling. `ensembles/blend_probs.py`
  sums probabilities across multiple probs csvs and argmaxes per row.
