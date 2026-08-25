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
