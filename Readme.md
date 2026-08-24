# Kaggle competition "Contradictory, My Dear Watson"

## Link

https://www.kaggle.com/competitions/contradictory-my-dear-watson

## Github

cmdw_kaggle

## Dataset

/Users/askash/kaggle_competitions/datasets/contradictory_my_dear_watson

See [CLAUDE.md](CLAUDE.md) for task/approach/plan details.

## Setup

```bash
mamba create -n cmdw python=3.10 -y
mamba activate cmdw
pip install -e .
```

## Run

All commands from repo root, env activated.

### EDA

Writes `experiments/eda_report.md` (versioned) + plots to `outputs/eda/`
(gitignored). Console log goes to stdout by
default; tee to a file to keep a copy:

```bash
python -m src.eda 2>&1 | tee outputs/eda/eda_run.log
```

### Smoke test (Mac, mps) — verify pipeline before real training

Tiny 100-row subset, 1 epoch, just checks the pipeline runs end to end
(checkpoint saved, `experiments/runs.md` row appended, submission validates):

```bash
python -m src.train --config configs/smoke.yaml 2>&1 | tee outputs/smoke_train.log
python -m src.infer --checkpoint outputs/checkpoints/<run_id>/best \
    --config configs/smoke.yaml \
    --out outputs/submissions/smoke.csv 2>&1 | tee outputs/smoke_infer.log
```

`<run_id>` printed by `train.py` at start of the run (also the dir name under
`outputs/checkpoints/`). `infer.py` tags the output filename with it
automatically (`--out smoke.csv` → `smoke_<run_id>.csv`), so a submission
file always traces back to the checkpoint that made it.

### Real training (GPU server, cuda)

```bash
python -m src.train --config configs/base.yaml 2>&1 | tee outputs/train_$(date +%Y%m%d_%H%M).log
python -m src.infer --checkpoint outputs/checkpoints/<run_id>/best \
    --config configs/base.yaml \
    --out outputs/submissions/sub_$(date +%Y%m%d_%H%M).csv
```

### ClearML (optional)

```bash
pip install -e .[clearml]
```

Set `clearml.enabled: true` in the config (plus `clearml.project_name`,
optionally `clearml.task_name`) and configure ClearML credentials as usual
(`clearml-init` or `~/clearml.conf`). Off by default — `src/clearml_logger.py`
isn't even imported into a running Task when disabled.

### Profiling

Set `profiling.enabled: true` in the config to capture a chrome trace of the
first `profiling.profile_steps` training steps to `outputs/profiling/<run_id>/`,
viewable with:

```bash
tensorboard --logdir outputs/profiling/<run_id>
```

then open the **PYTORCH_PROFILER** tab (needs `torch-tb-profiler`, already in
`pyproject.toml` deps). If `tensorboard` itself fails with
`ModuleNotFoundError: No module named 'pkg_resources'`, a too-new `setuptools`
(v81+) got installed over the pin — re-run `pip install -e .` or explicitly
`uv pip install "setuptools==75.6.0"`.

Alternative without tensorboard: load the `.pt.trace.json` file directly at
https://ui.perfetto.dev.
