"""Load train/test csv, stratified split, lazy-tokenizing torch Dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# 15 competition languages — matches base_cmdw lang_abv values exactly.
COMP_LANGS = {"ar", "bg", "zh", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi"}

# id column is unused downstream but present in base train/test for schema parity.
EXTRA_COLUMNS = ["premise", "hypothesis", "lang_abv", "label"]


def load_train_df(data_dir: Path, subset: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "train.csv")
    if subset is not None:
        df = df.sample(n=subset, random_state=0).reset_index(drop=True)
    return df


def load_test_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")


def load_multinli26lang_df(data_dir: Path, subsets: list[str]) -> pd.DataFrame:
    """Load translated-NLI parquet shards, e.g. `ar_mnli-00000-of-00001-<hash>.parquet`.

    Only shards whose `<lang>_<subset>` prefix matches a competition language
    and a requested subset (mnli, wanli, ...) are read — the corpus ships 26
    languages x 5 subsets and most combinations are out of scope here.
    """
    data_dir = Path(data_dir)
    frames = []
    for path in sorted(data_dir.glob("*.parquet")):
        lang, _, rest = path.stem.partition("_")
        subset = rest.split("-", 1)[0]
        if lang not in COMP_LANGS or subset not in subsets:
            continue
        shard = pd.read_parquet(path, columns=["premise", "hypothesis", "label"])
        shard["lang_abv"] = lang
        frames.append(shard)
    if not frames:
        logger.warning("no multilingual-NLI-26lang shards matched langs=%s subsets=%s", COMP_LANGS, subsets)
        return pd.DataFrame(columns=EXTRA_COLUMNS)
    return pd.concat(frames, ignore_index=True)[EXTRA_COLUMNS]


def load_xnli_df(data_dir: Path, splits: list[str]) -> pd.DataFrame:
    """Load xnli_hf parquet split(s) and explode the nested per-language
    columns into flat premise/hypothesis/lang_abv/label rows.

    Each source row bundles one premise/hypothesis pair translated into all
    15 languages: `premise` is a {lang: text} dict, `hypothesis` is
    {"language": [...], "translation": [...]} arrays aligned by position.
    """
    data_dir = Path(data_dir)
    frames = []
    for split in splits:
        shard_paths = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not shard_paths:
            raise FileNotFoundError(f"no xnli_hf parquet shards found for split '{split}' in {data_dir}")
        for path in shard_paths:
            raw = pd.read_parquet(path, columns=["premise", "hypothesis", "label"])
            for _, row in raw.iterrows():
                langs = row["hypothesis"]["language"]
                hyps = row["hypothesis"]["translation"]
                for lang, hyp in zip(langs, hyps):
                    if lang not in COMP_LANGS:
                        continue
                    frames.append(
                        {
                            "premise": row["premise"][lang],
                            "hypothesis": hyp,
                            "lang_abv": lang,
                            "label": row["label"],
                        }
                    )
    if not frames:
        logger.warning("no xnli_hf rows exploded for splits=%s", splits)
        return pd.DataFrame(columns=EXTRA_COLUMNS)
    return pd.DataFrame(frames, columns=EXTRA_COLUMNS)


def load_extra_train_df(cfg: dict[str, Any], exclude_pairs: set[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Assemble augmentation data from config-listed extra sources.

    Train-only augmentation: callers append this to the train split after
    `stratified_split` so validation stays a clean held-out slice of
    base_cmdw and per-language val accuracy isn't diluted by other corpora.

    `exclude_pairs` drops rows whose (premise, hypothesis) pair matches a
    pair already seen elsewhere — pass in the base test.csv (and/or train)
    pairs. EDA found 746 xnli_hf/multinli26lang rows that exactly duplicate
    test.csv pairs (same public source corpora), which would leak test
    labels into training without this filter.
    """
    extra_cfg = cfg.get("extra_data", {})

    frames = []
    multinli26_cfg = extra_cfg.get("multinli26lang")
    if multinli26_cfg and multinli26_cfg.get("enabled", False):
        frames.append(load_multinli26lang_df(Path(multinli26_cfg["data_dir"]), multinli26_cfg["subsets"]))
    xnli_cfg = extra_cfg.get("xnli_hf")
    if xnli_cfg and xnli_cfg.get("enabled", False):
        frames.append(load_xnli_df(Path(xnli_cfg["data_dir"]), xnli_cfg["splits"]))

    if not frames:
        return pd.DataFrame(columns=EXTRA_COLUMNS)
    df = pd.concat(frames, ignore_index=True)

    if exclude_pairs:
        before = len(df)
        keep = [pair not in exclude_pairs for pair in zip(df["premise"], df["hypothesis"])]
        df = df[keep].reset_index(drop=True)
        logger.info("dropped %d extra rows overlapping exclude_pairs (test/train leakage)", before - len(df))

    logger.info("loaded %d extra train rows from %d source(s)", len(df), len(frames))
    return df


def stratified_split(
    df: pd.DataFrame, val_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split stratified on (label, lang_abv) so validation mirrors train's
    class/language mix and per-language accuracy stays meaningful.

    sklearn requires every stratum to have >= 2 members. On the full dataset
    (label, lang) combos are big enough, but a tiny smoke-test subset can
    have singleton languages or labels — fall back through coarser keys
    (label+lang -> lang -> label -> no stratification) until it's valid.
    """
    candidates = [
        df["label"].astype(str) + "_" + df["lang_abv"],
        df["lang_abv"],
        df["label"].astype(str),
        None,
    ]
    for strata in candidates:
        if strata is None or strata.value_counts().min() >= 2:
            break
    train_df, val_df = train_test_split(
        df, test_size=val_size, random_state=seed, stratify=strata
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class NLIDataset(Dataset):
    """Tokenizes premise/hypothesis pair lazily in __getitem__ (not pre-batched
    in bulk) — keeps memory flat regardless of dataset size."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        has_labels: bool = True,
    ) -> None:
        self.premise    = df["premise"].to_numpy()
        self.hypothesis = df["hypothesis"].to_numpy()
        self.lang_abv   = df["lang_abv"].to_numpy()
        self.labels     = df["label"].to_numpy() if has_labels else None
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.premise)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Switched padding -> False from 'max_length'"""
        encoding = self.tokenizer(
            self.premise[idx],
            self.hypothesis[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            # return_tensors="pt",
        )
        item: dict[str, Any] = dict(encoding)
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item
