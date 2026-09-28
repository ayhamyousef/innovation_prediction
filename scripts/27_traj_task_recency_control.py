"""Year-only control on the trajectory-shape task.

scripts/25 asks how much of the predictability of the SEQUENCE-based labels follows
from emergence cohort rather than from the features. This script asks the same of the
trajectory-shape labels, the k=4 Euclidean k-means labeling that reproduces the
published construction and on which GBDT reaches the 0.831 reported in Section 5.6.
Same protocol as scripts/17 and 25: stratified 60/40, standardize on train, one-vs-rest
macro ROC-AUC, seed 42, only the predictor set varies.
"""
import json, numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ALL7=['ACCESS_SIZE','ACCESS_TREND','SIM_ACCESS','SIM_TECH','INVENT_DIVER','INVENT_APPL','ATTENT_SIZE']
df=pd.read_csv('results/clustering/technologies_labeled_all7_k3.csv')
y=np.load('results/paper_final_analysis/traj_labels_k4.npy')
assert len(df)==len(y), (len(df),len(y))
K=len(np.unique(y)); print("trajectory-shape labels: k=%d, sizes %s"%(K,np.bincount(y).tolist()))
out={}
for name,feats in [('all7',ALL7),('year_only',['emergence_year']),
                   ('no_access',[c for c in ALL7 if c not in ('ACCESS_SIZE','ACCESS_TREND')])]:
    X=df[feats].values.astype(np.float64)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.4,stratify=y,random_state=42)
    sc=StandardScaler(); Xtr=sc.fit_transform(Xtr).astype(np.float32); Xte=sc.transform(Xte).astype(np.float32)
    m=GradientBoostingClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,random_state=42).fit(Xtr,ytr)
    p,pr=m.predict(Xte),m.predict_proba(Xte)
    out[name]={'accuracy':round(float(accuracy_score(yte,p)),4),
               'macro_f1':round(float(f1_score(yte,p,average='macro')),4),
               'roc_auc_ovr_macro':round(float(roc_auc_score(yte,pr,multi_class='ovr',average='macro',labels=list(range(K)))),4)}
    print("  %-10s acc=%.4f macroF1=%.4f ROC-AUC=%.4f"%(name,out[name]['accuracy'],out[name]['macro_f1'],out[name]['roc_auc_ovr_macro']))
json.dump(out,open('results/paper_final_analysis/traj_task_recency_control.json','w'),indent=2)
print("\npaper reports 0.831 for GBDT all7 on this task")
