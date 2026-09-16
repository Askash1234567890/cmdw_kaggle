# Experiment log

Append-only. One row per training run, written automatically by `src/train.py`.
Never edit or delete a past row — compare by reading the table, not memory.

| date | run_id | model | lr | epochs | val_acc | val_acc_per_lang_notes | notes |
|------|--------|-------|----|--------|---------|------------------------|-------|
| 2026-08-24 | smoke_20260824_215902 | xlm-roberta-base | 2e-05 | 1 | 0.4000 | ar=0.00, fr=0.00, vi=0.00 |  |
| 2026-08-24 | smoke_20260824_220538 | xlm-roberta-base | 2e-05 | 1 | 0.4000 | ar=0.00, fr=0.00, vi=0.00 |  |
| 2026-08-25 | smoke_20260825_234347 | xlm-roberta-base | 2e-05 | 1 | 0.4000 | ar=0.00, fr=0.00, vi=0.00 |  |
| 2026-08-26 | base_20260826_184932 | xlm-roberta-base | 2e-05 | 50 | 0.7351 | de=0.58, fr=0.59, tr=0.63 |  |
| 2026-08-27 | large_20260827_231227 | xlm-roberta-large | 2e-05 | 50 | 0.8210 | fr=0.67, sw=0.69, zh=0.70 |  |
| 2026-09-07 | large_20260907_212128 | xlm-roberta-large | 2e-05 | 10 | 0.8193 | hi=0.65, fr=0.69, sw=0.72 |  |
| 2026-09-07 | large_20260907_221600 | xlm-roberta-large | 2e-05 | 5 | 0.3325 | vi=0.21, th=0.27, el=0.30 |  |
| 2026-09-08 | large_20260908_014145 | xlm-roberta-large | 2e-05 | 5 | 0.3259 | es=0.16, de=0.17, zh=0.17 |  |
| 2026-09-08 | large_20260908_020121 | xlm-roberta-large | 2e-05 | 35 | 0.3333 | sw=0.23, zh=0.25, ru=0.30 |  |
| 2026-09-08 | large_20260908_022820 | xlm-roberta-large | 1e-05 | 10 | 0.3498 | ru=0.22, tr=0.23, sw=0.31 |  |
| 2026-09-13 | large_20260913_163634 | xlm-roberta-large | 1e-05 | 3 | 0.3391 | hi=0.19, tr=0.23, bg=0.29 |  |
| 2026-09-13 | large_20260913_205921 | xlm-roberta-large | 1e-05 | 1 | 0.3292 | bg=0.15, th=0.22, sw=0.23 |  |
| 2026-09-14 | large_20260914_001600 | xlm-roberta-large | 1e-05 | 3 | 0.3622 | bg=0.24, tr=0.26, es=0.30 |  |
| 2026-09-15 | large_20260915_190515 | xlm-roberta-large | 1e-05 | 1 | 0.3350 | hi=0.24, de=0.25, bg=0.26 |  |
| 2026-09-16 | large_20260916_002327 | google/rembert | 1e-05 | 1 | 0.3201 | th=0.20, ru=0.26, ar=0.26 |  |
