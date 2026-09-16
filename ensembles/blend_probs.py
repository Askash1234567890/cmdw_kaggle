import pandas as pd

paths_to_probs = [
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260914_0852_large_20260913_205921_6M_data_1ep_seed42_score_0.8614_probs.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0003_large_20260915_190515_6M_data_1ep_seed1234_score_0.85601_probs.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0004_large_20260915_220023_6M_data_1ep_seed98765_score_0.85851_probs.csv",
    "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_20260916_0938_large_20260916_002327_google_seed42_score0.86621_probs.csv",
]

PROB_COLS = ["prob_0", "prob_1", "prob_2"]
OUT_PATH = "/home/boorble/askash/kaggle_comp/cmdw_kaggle/outputs/submissions/sub_blend_probs.csv"

dfs = [pd.read_csv(p).set_index("id") for p in paths_to_probs]

ids = dfs[0].index
for d in dfs[1:]:
    if not d.index.equals(ids):
        raise ValueError("id sets/order mismatch across probs files")

summed = sum(d[PROB_COLS] for d in dfs)

res = pd.DataFrame(index=ids)
res["prediction"] = summed[PROB_COLS].values.argmax(axis=1)
res = res.reset_index()
res.columns = ["id", "prediction"]

res.to_csv(OUT_PATH, index=False)
print(f"wrote {len(res)} rows to {OUT_PATH}")
print(res.head(10))
