"""Standalone EDA over train/test csv. Run: python -m src.eda --config configs/base.yaml

Writes plots + a markdown summary under the paths given in the config
(`paths.output_root`/eda, `paths.experiments_dir`).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import load_test_df, load_train_df
from src.utils import LABEL_NAMES, load_config, setup_logging

logger = logging.getLogger(__name__)


def _token_lengths(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Token count per premise/hypothesis via the configured model's tokenizer.
    Falls back to whitespace-split count if the tokenizer can't be
    downloaded (offline dev machine) — flagged in the report either way.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        premise_len = [len(tok.tokenize(t)) for t in df["premise"]]
        hyp_len = [len(tok.tokenize(t)) for t in df["hypothesis"]]
        method = f"{model_name} tokenizer"
    except Exception as e:  # offline / no network — degrade gracefully
        logger.warning("tokenizer load failed (%s), falling back to whitespace split", e)
        premise_len = [len(t.split()) for t in df["premise"]]
        hyp_len = [len(t.split()) for t in df["hypothesis"]]
        method = "whitespace split (tokenizer unavailable)"
    out = df.copy()
    out["premise_len"] = premise_len
    out["hyp_len"] = hyp_len
    out.attrs["length_method"] = method
    return out


def _plot_class_balance(df: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df["label"].map(LABEL_NAMES).value_counts().plot.bar(ax=axes[0], title="overall class balance")
    lang_label = df.groupby(["lang_abv", "label"]).size().unstack(fill_value=0)
    lang_label.plot.bar(stacked=True, ax=axes[1], title="class balance per language")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_length_dist(df_len: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_len["premise_len"], bins=40)
    axes[0].set_title("premise length (tokens)")
    axes[1].hist(df_len["hyp_len"], bins=40)
    axes[1].set_title("hypothesis length (tokens)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_lang_dist(train_df: pd.DataFrame, test_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    train_frac = train_df["lang_abv"].value_counts(normalize=True)
    test_frac = test_df["lang_abv"].value_counts(normalize=True)
    pd.DataFrame({"train": train_frac, "test": test_frac}).plot.bar(ax=ax)
    ax.set_title("language share: train vs test")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    dup_mask = df.duplicated(subset=["premise", "hypothesis"], keep=False)
    return df[dup_mask].sort_values(["premise", "hypothesis"])


def _label_spot_check(df: pd.DataFrame, n_per_cell: int = 1) -> pd.DataFrame:
    langs = sorted(df["lang_abv"].unique())
    rows = []
    for lang in langs:
        for label in (0, 1, 2):
            cell = df[(df["lang_abv"] == lang) & (df["label"] == label)]
            rows.append(cell.sample(n=min(n_per_cell, len(cell)), random_state=0))
    return pd.concat(rows) if rows else df.iloc[0:0]


def run_eda(cfg: dict[str, Any]) -> None:
    setup_logging()
    data_dir = Path(cfg["paths"]["data_dir"])
    out_dir = Path(cfg["paths"]["output_root"]) / "eda"
    report_dir = Path(cfg["paths"]["experiments_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_train_df(data_dir)
    test_df = load_test_df(data_dir)
    logger.info("train rows=%d test rows=%d", len(train_df), len(test_df))

    train_df_len = _token_lengths(train_df, cfg["model_name"])

    _plot_class_balance(train_df, out_dir / "class_balance.png")
    _plot_length_dist(train_df_len, out_dir / "length_dist.png")
    _plot_lang_dist(train_df, test_df, out_dir / "lang_dist_train_vs_test.png")

    dups = _find_duplicates(train_df)
    spot_check = _label_spot_check(train_df)
    spot_check.to_csv(out_dir / "label_spot_check.csv", index=False)

    train_langs = set(train_df["lang_abv"])
    test_langs = set(test_df["lang_abv"])
    lang_mismatch = train_langs.symmetric_difference(test_langs)

    class_counts = train_df["label"].map(LABEL_NAMES).value_counts()
    lang_counts = train_df["lang_abv"].value_counts()

    report = f"""# EDA report — Contradictory, My Dear Watson

Data: `{data_dir}`

## Shape
- train rows: {len(train_df)}
- test rows: {len(test_df)}

## Class balance (overall)
{class_counts.to_markdown()}

Roughly balanced (largest/smallest ratio {class_counts.max() / class_counts.min():.2f}) —
no resampling/class-weighting needed as a first pass.

## Language distribution (train)
{lang_counts.to_markdown()}

English dominates ({lang_counts.iloc[0]} rows, {lang_counts.iloc[0] / len(train_df) * 100:.1f}% of train),
the other 14 languages sit around 340-410 rows each. Low-resource languages
(bg, tr, de at the low end) are the ones most likely to show weak per-language
accuracy — watch these first in `experiments/runs.md`.

## Train vs test language mismatch
{"No language present in one split but not the other." if not lang_mismatch else f"Mismatch: {sorted(lang_mismatch)}"}

## Token length (method: {train_df_len.attrs['length_method']})
- premise: mean={train_df_len['premise_len'].mean():.1f}, p95={train_df_len['premise_len'].quantile(0.95):.0f}, max={train_df_len['premise_len'].max()}
- hypothesis: mean={train_df_len['hyp_len'].mean():.1f}, p95={train_df_len['hyp_len'].quantile(0.95):.0f}, max={train_df_len['hyp_len'].max()}
- combined p95 (premise+hypothesis, informs `max_length` in config): {(train_df_len['premise_len'] + train_df_len['hyp_len']).quantile(0.95):.0f}

## Duplicates
{len(dups)} row involved in an exact (premise, hypothesis) duplicate
({dups['label'].nunique() if len(dups) else 0} distinct label among them — a
duplicate pair with two different label would be a labeling conflict worth
checking manually).

## Label spot-check
One sample row per (language, label) written to `{out_dir / "label_spot_check.csv"}`
for manual read-through — no automated sanity check replaces eyeballing a few
real example per language.

## Plots
- `{out_dir / "class_balance.png"}`
- `{out_dir / "length_dist.png"}`
- `{out_dir / "lang_dist_train_vs_test.png"}`
"""
    report_path = report_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("EDA report written to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    run_eda(load_config(args.config))
