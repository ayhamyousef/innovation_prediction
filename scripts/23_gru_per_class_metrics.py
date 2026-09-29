"""
Per-class results for GBDT on the sequence-based (GRU) labels at k=3: precision,
recall, F1, one-vs-rest ROC-AUC and average precision for each class, under the
standard protocol (stratified 60/40 split, standardization fit on train, seed 42).

Usage (from the repository root):
    python scripts/23_gru_per_class_metrics.py

Requires the all-seven-feature table from 04b_recluster.py (run without --fae-k)
and the k=3 sequence labels from 17_gru_ksweep_downstream.py. Writes
results/paper_final_analysis/gru_k3_per_class_metrics.json.
"""
import numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support,
                             average_precision_score, classification_report)

FEATS = ['ACCESS_SIZE','ACCESS_TREND','SIM_ACCESS','SIM_TECH',
         'INVENT_DIVER','INVENT_APPL','ATTENT_SIZE']
# Sequence labels: use the k=3 labels produced by scripts/17, which are the ones the
# paper reports (cluster sizes 139,856 / 55,974 / 5,880). results/seq_autoencoder/
# cluster_labels.npy comes from an earlier run of scripts/16 and differs from these on
# 14 of 201,710 technologies; all reported results use the labels below.
LABELS = 'results/seq_autoencoder/ksweep/labels_k3.npy'

df = pd.read_csv('results/clustering/technologies_labeled_all7_k3.csv')
y = np.load(LABELS)
assert len(df) == len(y)
X = df[FEATS].values.astype(np.float64)

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
sc = StandardScaler(); Xtr = sc.fit_transform(Xtr).astype(np.float32); Xte = sc.transform(Xte).astype(np.float32)
m = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
m.fit(Xtr, ytr)
pred = m.predict(Xte); proba = m.predict_proba(Xte)

out = {}
out['accuracy'] = round(float(accuracy_score(yte, pred)), 4)
out['macro_f1'] = round(float(f1_score(yte, pred, average='macro')), 4)
out['roc_auc_ovr_macro'] = round(float(roc_auc_score(yte, proba, multi_class='ovr', average='macro', labels=[0,1,2])), 4)
p, r, f, s = precision_recall_fscore_support(yte, pred, labels=[0,1,2], zero_division=0)
out['per_class'] = {}
for i, name in enumerate(['S0','S1','S2']):
    out['per_class'][name] = {'precision': round(float(p[i]),4), 'recall': round(float(r[i]),4),
                              'f1': round(float(f[i]),4), 'support': int(s[i]),
                              'pr_auc': round(float(average_precision_score((yte==i).astype(int), proba[:,i])),4),
                              'roc_auc': round(float(roc_auc_score((yte==i).astype(int), proba[:,i])),4)}
out['train_counts'] = np.bincount(ytr).tolist()
out['test_counts']  = np.bincount(yte).tolist()
print(json.dumps(out, indent=2))
print(classification_report(yte, pred, labels=[0,1,2], target_names=['S0','S1','S2'], digits=4, zero_division=0))
json.dump(out, open('results/paper_final_analysis/gru_k3_per_class_metrics.json','w'), indent=2)
