"""Load train/test csv, stratified split, lazy-tokenizing torch Dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


def load_train_df(data_dir: Path, subset: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "train.csv")
    if subset is not None:
        df = df.sample(n=subset, random_state=0).reset_index(drop=True)
    return df


def load_test_df(data_dir: Path) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")


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
        self.premise = df["premise"].tolist()
        self.hypothesis = df["hypothesis"].tolist()
        self.lang_abv = df["lang_abv"].tolist()
        self.labels = df["label"].tolist() if has_labels else None
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.premise)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        encoding = self.tokenizer(
            self.premise[idx],
            self.hypothesis[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
