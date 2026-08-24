"""Fine-tune XLM-R on the NLI train set via HF Trainer.

Usage: python -m src.train --config configs/base.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from src.clearml_logger import close_clearml, init_clearml
from src.data import NLIDataset, load_train_df, stratified_split
from src.profiling import ProfilerCallback
from src.utils import get_device, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)

RUNS_LOG = Path("experiments/runs.md")


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    preds = np.argmax(eval_pred.predictions, axis=-1)
    return {"accuracy": accuracy_score(eval_pred.label_ids, preds)}


def per_language_accuracy(val_df: pd.DataFrame, preds: np.ndarray) -> dict[str, float]:
    per_lang = {}
    for lang, group in val_df.assign(pred=preds).groupby("lang_abv"):
        per_lang[lang] = accuracy_score(group["label"], group["pred"])
    return per_lang


def worst_languages_note(per_lang: dict[str, float], n: int = 3) -> str:
    worst = sorted(per_lang.items(), key=lambda kv: kv[1])[:n]
    return ", ".join(f"{lang}={acc:.2f}" for lang, acc in worst)


def append_run_row(
    run_id: str,
    model_name: str,
    lr: float,
    epochs: int,
    val_acc: float,
    per_lang_note: str,
    notes: str,
) -> None:
    date = datetime.now().strftime("%Y-%m-%d")
    row = (
        f"| {date} | {run_id} | {model_name} | {lr} | {epochs} | "
        f"{val_acc:.4f} | {per_lang_note} | {notes} |\n"
    )
    with open(RUNS_LOG, "a", encoding="utf-8") as f:
        f.write(row)
    logger.info("appended run row to %s", RUNS_LOG)


def build_run_id(run_name: str) -> str:
    return f"{run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def main(config_path: str) -> None:
    setup_logging()
    logger.info("stage: load config (%s)", config_path)
    cfg: dict[str, Any] = load_config(config_path)
    set_seed(cfg["seed"])
    device = get_device(cfg)
    logger.info("device=%s", device)

    run_id = build_run_id(cfg["run_name"])
    output_root = Path(cfg["output_root"])
    checkpoint_dir = output_root / "checkpoints" / run_id
    logger.info("run_id=%s", run_id)

    logger.info("stage: init clearml (enabled=%s)", cfg.get("clearml", {}).get("enabled", False))
    clearml_task = init_clearml(cfg, run_id)

    logger.info("stage: load + split train.csv")
    df = load_train_df(subset=cfg["data"]["subset"])
    train_df, val_df = stratified_split(df, val_size=cfg["data"]["val_size"], seed=cfg["seed"])
    logger.info("train=%d val=%d", len(train_df), len(val_df))

    logger.info("stage: load tokenizer %s (downloads on first run)", cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    train_dataset = NLIDataset(train_df, tokenizer, cfg["max_length"])
    val_dataset = NLIDataset(val_df, tokenizer, cfg["max_length"])

    logger.info(
        "stage: load model %s (~2.2GB download on first run, cached after in ~/.cache/huggingface)",
        cfg["model_name"],
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=cfg["num_labels"]
    )
    logger.info("stage: move model to device=%s", device)
    model = model.to(device)

    t = cfg["training"]
    args = TrainingArguments(
        output_dir=str(checkpoint_dir / "hf_run"),
        num_train_epochs=t["epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["eval_batch_size"],
        learning_rate=t["lr"],
        weight_decay=t["weight_decay"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        bf16=t["bf16"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        save_strategy=t["save_strategy"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        seed=cfg["seed"],
        report_to=["tensorboard"],
        no_cuda=(cfg["device"] == "cpu"),
    )

    callbacks = []
    if cfg["profiling"]["enabled"]:
        profiling_dir = output_root / "profiling" / run_id
        callbacks.append(ProfilerCallback(profiling_dir, cfg["profiling"]["profile_steps"]))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )
    logger.info("stage: trainer built, starting train()")

    try:
        trainer.train()
        logger.info("stage: train() done, running final eval on val_dataset")

        predictions = trainer.predict(val_dataset)
        preds = np.argmax(predictions.predictions, axis=-1)
        val_acc = accuracy_score(val_df["label"], preds)
        per_lang = per_language_accuracy(val_df, preds)
        per_lang_note = worst_languages_note(per_lang)
        logger.info("val_acc=%.4f worst_languages=%s", val_acc, per_lang_note)
        for lang, acc in sorted(per_lang.items(), key=lambda kv: kv[1]):
            logger.info("  %s: %.4f", lang, acc)

        logger.info("stage: saving checkpoint")
        best_dir = checkpoint_dir / "best"
        trainer.save_model(str(best_dir))
        tokenizer.save_pretrained(str(best_dir))
        shutil.copy(config_path, checkpoint_dir / "config.yaml")
        logger.info("checkpoint saved to %s", best_dir)
    finally:
        close_clearml(clearml_task)

    append_run_row(
        run_id=run_id,
        model_name=cfg["model_name"],
        lr=t["lr"],
        epochs=t["epochs"],
        val_acc=val_acc,
        per_lang_note=per_lang_note,
        notes=cfg.get("notes", ""),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
