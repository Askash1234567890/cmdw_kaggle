"""Predict class probabilities on test.csv from a trained checkpoint.

Usage: python -m src.infer_probs --checkpoint outputs/checkpoints/<run_id>/best \
    --config configs/base.yaml --out outputs/submissions/probs.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src.data import NLIDataset, load_test_df
from src.utils import get_device, load_config, setup_logging

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 5195
PROB_DECIMALS = 6


def predict_probs(checkpoint: str, cfg: dict[str, Any]) -> pd.DataFrame:
    device = get_device(cfg)
    logger.info("device=%s", device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    test_df = load_test_df(Path(cfg["paths"]["data_dir"]))
    dataset = NLIDataset(test_df, tokenizer, cfg["max_length"], has_labels=False)
    loader = DataLoader(
        dataset,
        batch_size = cfg["training"]["eval_batch_size"],
        collate_fn = DataCollatorWithPadding(tokenizer),
        shuffle    = False)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.concatenate(all_probs, axis=0)
    out = pd.DataFrame(probs_arr, columns=["prob_0", "prob_1", "prob_2"])
    out.insert(0, "id", test_df["id"].values)
    return out


def validate_probs(probs: pd.DataFrame, data_dir: Path) -> None:
    errors = []

    if len(probs) != EXPECTED_ROWS:
        errors.append(f"row count {len(probs)} != {EXPECTED_ROWS}")

    if list(probs.columns) != ["id", "prob_0", "prob_1", "prob_2"]:
        errors.append(f"columns {list(probs.columns)} != ['id', 'prob_0', 'prob_1', 'prob_2']")

    prob_cols = probs[["prob_0", "prob_1", "prob_2"]]
    if prob_cols.isna().any().any():
        errors.append("probabilities contain NaN")

    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    if set(probs["id"]) != set(sample["id"]):
        errors.append("id set does not match sample_submission.csv id set")

    if errors:
        raise ValueError("probs output failed checklist:\n" + "\n".join(f"- {e}" for e in errors))


def run_id_from_checkpoint(checkpoint: str) -> str:
    """Checkpoint path is outputs/checkpoints/<run_id>/best — pull run_id
    back out so outputs stay traceable to the model that made them."""
    return Path(checkpoint).resolve().parent.name


def tag_out_path(out_path: str, run_id: str) -> Path:
    out = Path(out_path)
    return out.with_name(f"{out.stem}_{run_id}{out.suffix}")


def main(checkpoint: str, config_path: str, out_path: str) -> None:
    setup_logging()
    cfg = load_config(config_path)
    run_id = run_id_from_checkpoint(checkpoint)

    probs = predict_probs(checkpoint, cfg)
    prob_cols = ["prob_0", "prob_1", "prob_2"]
    probs[prob_cols] = probs[prob_cols].round(PROB_DECIMALS)
    validate_probs(probs, Path(cfg["paths"]["data_dir"]))
    logger.info("probs passed checklist (%d rows)", len(probs))

    out = tag_out_path(out_path, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    probs.to_csv(out, index=False)
    logger.info("probs written to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.out)
