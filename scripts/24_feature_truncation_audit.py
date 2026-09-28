"""Audit the left-truncation of ACCESS_SIZE and ACCESS_TREND.

Both features sum component patent counts over the five years before a
technology emerges. The assembled patent record starts in 2002, so for the
2002 cohort that window lies entirely outside the data and the features are
identically zero; they stay undercounted until the 2007 cohort, the first with
a fully covered window. This script quantifies the effect and reports the
cluster contrasts with and without the affected cohorts. Backs the fifth
limitation in Section 6.2 and the qualification in Section 5.2.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CSV = "results/clustering/technologies_labeled_fae_k3_k3.csv"
OUT = "results/paper_final_analysis/feature_truncation_audit.json"
WINDOW, DATA_START = 5, 2002
FIRST_COMPLETE = DATA_START + WINDOW          # 2007

df = pd.read_csv(CSV)
out = {"n": int(len(df)), "data_start": DATA_START, "access_window": WINDOW,
       "first_complete_cohort": FIRST_COMPLETE}

by_year = df.groupby("emergence_year")["ACCESS_SIZE"]
out["by_year"] = {int(y): {"n": int(by_year.size()[y]),
                           "mean_access_size": round(float(by_year.mean()[y]), 1),
                           "pct_exactly_zero": round(float((df[df.emergence_year == y].ACCESS_SIZE == 0).mean() * 100), 1)}
                  for y in sorted(df.emergence_year.unique())[:8]}

affected = df[df.emergence_year < FIRST_COMPLETE]
out["affected_n"] = int(len(affected))
out["affected_pct"] = round(100 * len(affected) / len(df), 1)
out["spearman_access_size_year"] = round(float(spearmanr(df.ACCESS_SIZE, df.emergence_year)[0]), 3)
out["spearman_access_trend_year"] = round(float(spearmanr(df.ACCESS_TREND, df.emergence_year)[0]), 3)

def cluster_table(d):
    return {int(c): {"n": int(len(g)),
                     "median_year": int(g.emergence_year.median()),
                     "mean_access_size": round(float(g.ACCESS_SIZE.mean()), 1)}
            for c, g in d.groupby("cluster")}

out["clusters_full"] = cluster_table(df)
out["clusters_complete_window_only"] = cluster_table(df[df.emergence_year >= FIRST_COMPLETE])
for key in ("clusters_full", "clusters_complete_window_only"):
    m = sorted(v["mean_access_size"] for v in out[key].values())
    out[key + "_ratio_range"] = [round(m[-1] / m[1], 1), round(m[-1] / m[0], 1)]

json.dump(out, open(OUT, "w"), indent=2)
print(json.dumps(out, indent=2))
print(f"\nfull corpus ratio range      : {out['clusters_full_ratio_range']}")
print(f"complete-window ratio range  : {out['clusters_complete_window_only_ratio_range']}")
