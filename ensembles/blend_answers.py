import pandas as pd

paths_to_answers = [
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260914_0852_large_20260913_205921_6M_data_1ep_seed42_score_0.8614.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0003_large_20260915_190515_6M_data_1ep_seed1234_score_0.85601.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0004_large_20260915_220023_6M_data_1ep_seed98765_score_0.85851.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0938_large_20260916_002327_google_seed42_score0.86621.csv",
]
# best single model, used as tie-breaker for even-vote deadlocks
BEST_MODEL_IDX = 3
OUT_PATH = "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_blend_mode.csv"

df = pd.DataFrame()
for i, path in enumerate(paths_to_answers):
    df_cur = pd.read_csv(path)
    df[f"ans{i}"] = df_cur["prediction"]
df["id"] = df_cur["id"]
df.set_index("id", inplace=True)

pred_cols = [f"ans{i}" for i in range(len(paths_to_answers))]
modes = df[pred_cols].mode(axis=1)  # NaN in extra cols when no tie

def pick(row: pd.Series) -> int:
    row_modes = row.dropna()
    if len(row_modes) == 1:
        return int(row_modes.iloc[0])
    return int(df.loc[row.name, f"ans{BEST_MODEL_IDX}"])

res = modes.apply(pick, axis=1)
res = res.reset_index()
res.columns = ["id", "prediction"]

res.to_csv(OUT_PATH, index=False)
print(f"wrote {len(res)} rows to {OUT_PATH}")
print(res.head(10))
