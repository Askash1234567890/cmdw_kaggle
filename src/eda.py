"""Standalone EDA over train/test csv. Run: python -m src.eda

Writes plots + a markdown summary to outputs/eda/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import DATA_DIR, load_test_df, load_train_df
from src.utils import LABEL_NAMES, setup_logging

logger = logging.getLogger(__name__)

OUT_DIR = Path("outputs/eda")
REPORT_DIR = Path("experiments")
MODEL_NAME = "xlm-roberta-large"


def _token_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Token count per premise/hypothesis via the XLM-R tokenizer.
    Falls back to whitespace-split count if the tokenizer can't be
    downloaded (offline dev machine) — flagged in the report either way.
    """
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        premise_len = [len(tok.tokenize(t)) for t in df["premise"]]
        hyp_len = [len(tok.tokenize(t)) for t in df["hypothesis"]]
        method = f"{MODEL_NAME} tokenizer"
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


def run_eda() -> None:
    setup_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_train_df()
    test_df = load_test_df()
    logger.info("train rows=%d test rows=%d", len(train_df), len(test_df))

    train_df_len = _token_lengths(train_df)

    _plot_class_balance(train_df, OUT_DIR / "class_balance.png")
    _plot_length_dist(train_df_len, OUT_DIR / "length_dist.png")
    _plot_lang_dist(train_df, test_df, OUT_DIR / "lang_dist_train_vs_test.png")

    dups = _find_duplicates(train_df)
    spot_check = _label_spot_check(train_df)
    spot_check.to_csv(OUT_DIR / "label_spot_check.csv", index=False)

    train_langs = set(train_df["lang_abv"])
    test_langs = set(test_df["lang_abv"])
    lang_mismatch = train_langs.symmetric_difference(test_langs)

    class_counts = train_df["label"].map(LABEL_NAMES).value_counts()
    lang_counts = train_df["lang_abv"].value_counts()

    report = f"""# EDA report — Contradictory, My Dear Watson

Data: `{DATA_DIR}`

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
One sample row per (language, label) written to `outputs/eda/label_spot_check.csv`
for manual read-through — no automated sanity check replaces eyeballing a few
real example per language.

## Plots
- `outputs/eda/class_balance.png`
- `outputs/eda/length_dist.png`
- `outputs/eda/lang_dist_train_vs_test.png`
"""
    report_path = REPORT_DIR / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("EDA report written to %s", report_path)


if __name__ == "__main__":
    run_eda()
