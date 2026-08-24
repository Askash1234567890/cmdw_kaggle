# Contradictory, My Dear Watson — NLI competition

## Task

Classify premise/hypothesis pair into 3 class:
- 0 = entailment
- 1 = neutral
- 2 = contradiction

15 language in train/test (ar, bg, zh, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi). Metric: accuracy.

Submission: csv, exact 5195 row + header, columns `id,prediction`.

## Data

Local path (not in repo, do not commit):
`/Users/askash/kaggle_competitions/datasets/contradictory_my_dear_watson/`
- `train.csv` — id, premise, hypothesis, lang_abv, language, label
- `test.csv` — same minus label
- `sample_submission.csv`

Reference by this absolute path in code/configs, never copy into repo.

## Approach

Fine-tune `xlm-roberta-large` on train.csv using torch + transformers `Trainer` (not hand-rolled loop, not Lightning). No TPU/Kaggle-notebook path — dev + train happen locally, portable via mamba env.

Profiling hooks built into training pipeline from the start (`src/profiling.py`, config-gated `torch.profiler` callback) — used later to find and fix training bottleneck, off by default.

Optional ClearML tracking lives in its own module (`src/clearml_logger.py`, not mixed into `train.py`), gated by `clearml.enabled` in config, off by default. Optional dep: `pip install -e .[clearml]`.

## Hardware

- **Mac M5 (local dev machine)**: smoke test only — tiny data subset, `configs/smoke.yaml`, `device: mps`. Verifies pipeline mechanics, not real accuracy.
- **RTX 5090 (GPU server, CUDA 13.08, driver 580)**: real training, `configs/base.yaml`, `device: cuda`.
- Device is set explicitly in the yaml config, never auto-detected — swap `device:` manually when moving between machine.

## Repo structure (target)

```
contradictory_my_dear_watson/
  CLAUDE.md
  Readme.md
  pyproject.toml         # env + deps, single source of truth
  src/
    data.py               # load + preprocess train/test
    eda.py                 # standalone EDA script/functions
    train.py                # fine-tune loop
    infer.py                 # predict on test.csv, write submission
    utils.py                  # seeding, metrics, small shared helper
  configs/
    base.yaml               # model name, lr, epochs, batch size, seed
  notebooks/                 # exploration only, never pipeline logic
  experiments/
    runs.md                  # append-only log, one row per run
  outputs/
    checkpoints/               # gitignored
    submissions/                # gitignored, timestamped csv
    eda/                          # gitignored, plots/reports from eda.py
```

## Environment (mamba, pyproject.toml)

Portability to GPU server matters — env spec lives in `pyproject.toml`, mamba env just wraps python+cuda.

```bash
mamba create -n cmdw python=3.10 -y
mamba activate cmdw
pip install -e .
```

`pyproject.toml` pin exact version for torch/transformers/datasets/scikit-learn/pandas — no bare `>=`. Bump deliberately, note in commit message why.

Check `torch.cuda.is_available()` before training. CPU fallback only for smoke test on tiny subset (e.g. 100 row).

## Commands (once code exist)

```bash
python -m src.eda
python -m src.train --config configs/base.yaml
python -m src.infer --checkpoint outputs/checkpoints/best --out outputs/submissions/sub_$(date +%Y%m%d_%H%M).csv
```

## Code style — senior level

- Type hints on every function signature, no bare `Any` unless truly dynamic.
- No premature abstraction — three similar lines beat a wrong helper. Only extract when a second real caller exist.
- Small, single-purpose functions. `train.py` main loop stays readable top to bottom — no hidden control flow buried in callback.
- No silent exception swallow. No bare `except:`.
- No dead code, no commented-out block, no `# TODO` without owner/reason.
- Config-driven, not hardcoded — model name, lr, seed, paths all in `configs/*.yaml`, never inline magic value in `train.py`.
- Deterministic — seed torch/numpy/random/dataloader every run, log the seed used.
- Docstring only where behavior non-obvious (e.g. why a specific tokenization quirk handled). Skip docstring that just restate the signature.
- Prefer `pathlib.Path` over string path concat.
- Logging via `logging` module, not scattered `print`.

## EDA (before first training run, and after any data assumption change)

`src/eda.py` should answer, report saved to `experiments/eda_report.md` (versioned), plots to `outputs/eda/` (gitignored):
- class balance overall and per language
- premise/hypothesis length distribution (token count via chosen tokenizer, not just chars)
- language distribution in train vs test — flag mismatch
- duplicate or near-duplicate premise/hypothesis pair
- any label noise sample (spot-check few row per class per language)

Findings feed decision in `experiments/runs.md`, not just left in notebook.

## Experiment tracking / version comparison

Append-only table in `experiments/runs.md`, one row per training run:

```
| date       | run_id | model            | lr    | epochs | val_acc | val_acc_per_lang_notes         | notes                        |
|------------|--------|------------------|-------|--------|---------|---------------------------------|-------------------------------|
| 2026-08-24 | r001   | xlm-roberta-base | 2e-5  | 3      | 0.812   | weak on sw, th                  | baseline, stratified split    |
```

Rule:
- every run get a `run_id`, checkpoint saved as `outputs/checkpoints/<run_id>/`.
- config file for that run copied alongside checkpoint (`outputs/checkpoints/<run_id>/config.yaml`) so run fully reproducible from artifact alone.
- never overwrite a previous run's row — append only, compare by reading table not memory.
- when trying new idea (different model, loss, augmentation), state hypothesis in `notes` before running, result after.

## Plan

- [x] `pyproject.toml` + `.gitignore`
- [x] `src/eda.py` — run on real data, findings logged below
- [x] `src/utils.py` — seed, config loader, device resolver, logging
- [x] `src/data.py` — stratified split, `NLIDataset`
- [x] `src/profiling.py` — config-gated `torch.profiler` callback
- [x] `configs/base.yaml` + `configs/smoke.yaml`
- [x] `src/train.py` — HF Trainer, per-language eval, auto-append `experiments/runs.md`
- [x] `src/infer.py` — predict + submission checklist validation
- [x] Smoke test on Mac (M5, mps, tiny subset) — user ran, pipeline verified end to end (train → eval → checkpoint → runs.md row), plus profiling hook verified
- [ ] Real training run on 5090 (`configs/base.yaml`, device: cuda) — done by user

## Changelog

- 2026-08-24 — plan written, repo scaffolding started.
- 2026-08-24 — EDA run on real data (12120 train / 5195 test rows). Findings:
  class balance near-even (ratio 1.08, no resampling needed); English 56.7%
  of train, other 14 languages ~340-410 rows each (bg/tr/de smallest — watch
  these for weak per-language accuracy); no train/test language mismatch; 0
  exact-duplicate (premise, hypothesis) pairs; combined premise+hypothesis
  p95 token length (xlm-roberta-large tokenizer) = 74 → `max_length: 128` set
  in `configs/base.yaml`. Full report: `experiments/eda_report.md`.
- 2026-08-24 — pipeline code written: `src/{utils,data,eda,profiling,train,infer}.py`,
  `configs/{base,smoke}.yaml`, `experiments/runs.md` initialized. Smoke test
  left for user to run locally (instructions in Readme.md) rather than run
  from this session.
- 2026-08-24 — EDA report path moved to `experiments/eda_report.md` (versioned),
  plots/csv stay `outputs/eda/` (gitignored).
- 2026-08-24 — optional ClearML logging added (`src/clearml_logger.py`,
  `clearml.enabled` flag in config, off by default, separate optional dep).
- 2026-08-24 — smoke test run by user on Mac (mps, `xlm-roberta-base` for
  speed/RAM — `smoke.yaml` deliberately uses a smaller model than `base.yaml`'s
  `xlm-roberta-large`). Pipeline verified end to end. Missing deps found and
  pinned in `pyproject.toml`: `accelerate` (HF Trainer hard requirement, was
  missing entirely), `setuptools==75.6.0` pinned (v81+ dropped `pkg_resources`,
  which `tensorboard`'s CLI needs at runtime), `torch-tb-profiler` (needed for
  tensorboard's PYTORCH_PROFILER tab to read `torch.profiler` trace.json —
  without it tensorboard shows "no dashboards active").
- 2026-08-24 — read profiler trace.json manually (tensorboard UI trace-tree
  parser threw non-fatal errors on the mps trace): `Optimizer.step#AdamW.step`
  was ~94% of profiled step time, forward/backward (`scaled_dot_product_attention`)
  negligible by comparison — AdamW likely running non-fused/non-foreach on mps,
  one kernel launch per parameter. Mac-only artifact, not worth fixing — 5090
  (cuda) run should default to fused AdamW. Re-profile on real `base.yaml` run
  on the GPU server, not off the smoke test, before drawing conclusions there.

## Submission checklist

- [ ] row count == 5195 + header
- [ ] columns exactly `id,prediction`
- [ ] prediction in {0,1,2}, int not float
- [ ] id set match sample_submission.csv id set exactly
- [ ] no NaN prediction

## Conventions

- seed everything (torch, numpy, python random) for reproducibility
- log val accuracy per epoch, keep best checkpoint by val accuracy
- don't leak val language distribution — stratify split by (label, lang_abv)
- gitignore: `outputs/checkpoints/`, `outputs/submissions/*.csv`, `outputs/eda/`, `*.pt`, `*.bin`, mamba env not gitignored (env not stored in repo at all)
