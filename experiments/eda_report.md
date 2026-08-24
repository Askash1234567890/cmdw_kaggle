# EDA report — Contradictory, My Dear Watson

Data: `/Users/askash/kaggle_competitions/datasets/contradictory_my_dear_watson`

## Shape
- train rows: 12120
- test rows: 5195

## Class balance (overall)
| label         |   count |
|:--------------|--------:|
| entailment    |    4176 |
| contradiction |    4064 |
| neutral       |    3880 |

Roughly balanced (largest/smallest ratio 1.08) —
no resampling/class-weighting needed as a first pass.

## Language distribution (train)
| lang_abv   |   count |
|:-----------|--------:|
| en         |    6870 |
| zh         |     411 |
| ar         |     401 |
| fr         |     390 |
| sw         |     385 |
| ur         |     381 |
| vi         |     379 |
| ru         |     376 |
| hi         |     374 |
| el         |     372 |
| th         |     371 |
| es         |     366 |
| tr         |     351 |
| de         |     351 |
| bg         |     342 |

English dominates (6870 rows, 56.7% of train),
the other 14 languages sit around 340-410 rows each. Low-resource languages
(bg, tr, de at the low end) are the ones most likely to show weak per-language
accuracy — watch these first in `experiments/runs.md`.

## Train vs test language mismatch
No language present in one split but not the other.

## Token length (method: xlm-roberta-base tokenizer)
- premise: mean=27.4, p95=56, max=226
- hypothesis: mean=13.6, p95=24, max=58
- combined p95 (premise+hypothesis, informs `max_length` in config): 74

## Duplicates
0 row involved in an exact (premise, hypothesis) duplicate
(0 distinct label among them — a
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
