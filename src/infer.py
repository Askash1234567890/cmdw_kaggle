"""Predict on test.csv from a trained checkpoint, write + validate submission.

Usage: python -m src.infer --checkpoint outputs/checkpoints/<run_id>/best \
    --config configs/base.yaml --out outputs/submissions/sub.csv
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import NLIDataset, load_test_df
from src.utils import get_device, load_config, setup_logging

logger = logging.getLogger(__name__)

EXPECTED_ROWS = 5195
VALID_LABELS = {0, 1, 2}


def predict(checkpoint: str, cfg: dict[str, Any]) -> pd.DataFrame:
    device = get_device(cfg)
    logger.info("device=%s", device)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    model.eval()

    test_df = load_test_df(Path(cfg["paths"]["data_dir"]))
    dataset = NLIDataset(test_df, tokenizer, cfg["max_length"], has_labels=False)
    loader = DataLoader(dataset, batch_size=cfg["training"]["eval_batch_size"], shuffle=False)

    all_preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())

    return pd.DataFrame({"id": test_df["id"], "prediction": all_preds})


def validate_submission(sub: pd.DataFrame, data_dir: Path) -> None:
    errors = []

    if len(sub) != EXPECTED_ROWS:
        errors.append(f"row count {len(sub)} != {EXPECTED_ROWS}")

    if list(sub.columns) != ["id", "prediction"]:
        errors.append(f"columns {list(sub.columns)} != ['id', 'prediction']")

    if sub["prediction"].isna().any():
        errors.append("prediction contains NaN")

    non_int = sub["prediction"].dropna().apply(lambda x: float(x) != int(x))
    if non_int.any():
        errors.append("prediction contains non-integer values")

    bad_labels = set(sub["prediction"].dropna().astype(int).unique()) - VALID_LABELS
    if bad_labels:
        errors.append(f"prediction contains labels outside {VALID_LABELS}: {bad_labels}")

    sample = pd.read_csv(Path(data_dir) / "sample_submission.csv")
    if set(sub["id"]) != set(sample["id"]):
        errors.append("id set does not match sample_submission.csv id set")

    if errors:
        raise ValueError("submission failed checklist:\n" + "\n".join(f"- {e}" for e in errors))


def run_id_from_checkpoint(checkpoint: str) -> str:
    """Checkpoint path is outputs/checkpoints/<run_id>/best — pull run_id
    back out so submissions stay traceable to the model that made them."""
    return Path(checkpoint).resolve().parent.name


def tag_out_path(out_path: str, run_id: str) -> Path:
    out = Path(out_path)
    return out.with_name(f"{out.stem}_{run_id}{out.suffix}")


def main(checkpoint: str, config_path: str, out_path: str) -> None:
    setup_logging()
    cfg = load_config(config_path)
    run_id = run_id_from_checkpoint(checkpoint)

    sub = predict(checkpoint, cfg)
    sub["prediction"] = sub["prediction"].astype(int)
    validate_submission(sub, Path(cfg["paths"]["data_dir"]))
    logger.info("submission passed checklist (%d rows)", len(sub))

    out = tag_out_path(out_path, run_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    logger.info("submission written to %s", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.checkpoint, args.config, args.out)
