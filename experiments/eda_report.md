# EDA report — Contradictory, My Dear Watson

Data: `/home/boorble/askash/datasets/kaggle/cmdw_data/base_cmdw` (base) + extra_data config sources below.

## Shape
- base train rows: 12120
- test rows: 5195
- extra rows (augmentation corpora): 6477880
- combined train-pool rows: 6490000

## Source breakdown (combined)
| source         |            count |
|:---------------|-----------------:|
| xnli_hf        |      5.92788e+06 |
| multinli26lang | 550000           |
| base_cmdw      |  12120           |

## Class balance (combined)
| label         |       count |
|:--------------|------------:|
| neutral       | 2.20226e+06 |
| entailment    | 2.17464e+06 |
| contradiction | 2.1131e+06  |

Largest/smallest ratio 1.04 —
roughly balanced, no resampling/class-weighting needed.

## Language distribution (combined train-pool)
| lang_abv   |   count |
|:-----------|--------:|
| zh         |  445603 |
| ar         |  445593 |
| fr         |  445582 |
| sw         |  445577 |
| ur         |  445573 |
| vi         |  445571 |
| ru         |  445568 |
| hi         |  445566 |
| es         |  445558 |
| de         |  445543 |
| tr         |  445543 |
| en         |  402062 |
| el         |  395564 |
| th         |  395563 |
| bg         |  395534 |

## Train vs test language mismatch
No language present in one split but not the other.

## Token length (method: xlm-roberta-large tokenizer, sampled up to 50,000 rows from combined pool)
- premise: mean=32.2, p95=73, max=510
- hypothesis: mean=16.5, p95=31, max=101
- combined p95 (premise+hypothesis, informs `max_length` in config): 97

## Duplicates (base train only)
0 row involved in an exact (premise, hypothesis) duplicate
(0 distinct label among them — a
duplicate pair with two different label would be a labeling conflict worth
checking manually).

## Extra-data leakage check
- extra rows exactly matching a (premise, hypothesis) pair in **test.csv**: 746
- extra rows exactly matching a (premise, hypothesis) pair in **base train.csv**: 1744

WARNING: extra_data overlaps test.csv pairs — drop these rows before training or val/test accuracy will be inflated by leakage.
Note: 1744 extra rows duplicate base train rows — harmless but adds no signal, consider deduping.

## Label spot-check
One sample row per (language, label) from the combined pool, written to
`outputs/eda/label_spot_check.csv` for manual read-through — no automated
sanity check replaces eyeballing a few real example per language.

## Plots
- `outputs/eda/class_balance.png`
- `outputs/eda/length_dist.png`
- `outputs/eda/lang_dist_train_vs_test.png`
