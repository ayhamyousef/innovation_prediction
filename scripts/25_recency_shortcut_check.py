"""Is the sequence-label prediction a recency shortcut?

ACCESS_SIZE is left-truncated for pre-2007 cohorts and ranks 0.87 with emergence
year (scripts/24), and the S1 cluster is dominated by recent, right-censored
technologies. It is therefore fair to ask whether GBDT's 0.914 on the sequence
labels is partly the model detecting how late a technology appeared. Three
controls, same protocol as scripts/17 (stratified 60/40, standardize on train,
OvR macro ROC-AUC, seed 42):

  all7            the reported configuration
  no_access       drop ACCESS_SIZE and ACCESS_TREND, the two truncated features
  year_only       emergence year as the single predictor
  all7_plus_year  all seven features plus emergence year
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ALL7 = ['ACCESS_SIZE', 'ACCESS_TREND', 'SIM_ACCESS', 'SIM_TECH',
        'INVENT_DIVER', 'INVENT_APPL', 'ATTENT_SIZE']
TRUNCATED = ['ACCESS_SIZE', 'ACCESS_TREND']

# Sequence labels: use the k=3 labels produced by scripts/17, which are the ones the
# paper reports (cluster sizes 139,856 / 55,974 / 5,880). results/seq_autoencoder/
# cluster_labels.npy comes from an earlier run of scripts/16 and differs from these on
# 14 of 201,710 technologies; all reported results use the labels below.
LABELS = 'results/seq_autoencoder/ksweep/labels_k3.npy'

df = pd.read_csv('results/clustering/technologies_labeled_all7_k3.csv')
y = np.load(LABELS)
assert len(df) == len(y)

configs = {
    'all7':           ALL7,
    'no_access':      [c for c in ALL7 if c not in TRUNCATED],
    'year_only':      ['emergence_year'],
    'all7_plus_year': ALL7 + ['emergence_year'],
}

rows = {}
for name, feats in configs.items():
    X = df[feats].values.astype(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr).astype(np.float32)
    Xte = sc.transform(Xte).astype(np.float32)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                   learning_rate=0.1, random_state=42).fit(Xtr, ytr)
    pred, proba = m.predict(Xte), m.predict_proba(Xte)
    rows[name] = {
        'n_features': len(feats),
        'accuracy': round(float(accuracy_score(yte, pred)), 4),
        'macro_f1': round(float(f1_score(yte, pred, average='macro')), 4),
        'roc_auc_ovr_macro': round(float(roc_auc_score(yte, proba, multi_class='ovr',
                                                       average='macro', labels=[0, 1, 2])), 4),
    }
    print(f"  {name:16s} k={len(feats)}  acc={rows[name]['accuracy']:.4f}  "
          f"macroF1={rows[name]['macro_f1']:.4f}  ROC-AUC={rows[name]['roc_auc_ovr_macro']:.4f}")

json.dump(rows, open('results/paper_final_analysis/recency_shortcut_check.json', 'w'), indent=2)
print("\nreported all7            :", rows['all7']['roc_auc_ovr_macro'])
print("without truncated feats  :", rows['no_access']['roc_auc_ovr_macro'])
print("emergence year alone     :", rows['year_only']['roc_auc_ovr_macro'])
print("all7 + emergence year    :", rows['all7_plus_year']['roc_auc_ovr_macro'])
