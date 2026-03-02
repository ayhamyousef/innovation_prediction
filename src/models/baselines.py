"""
Baseline models for comparison:
  1. TF-IDF + Logistic Regression (text-only)
  2. Metadata-only classifiers (LR, RF, GBDT)
  3. TF-IDF + Metadata combined

These replicate and extend the baselines from the Chen et al. paper.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_auc_score, log_loss
)
from scipy.sparse import hstack, csr_matrix

logger = logging.getLogger("innovation_prediction.baselines")


# ============================================================
# TF-IDF + Logistic Regression
# ============================================================

class TFIDFBaseline:
    """TF-IDF vectorizer + Logistic Regression classifier."""

    def __init__(self, tfidf_cfg: Dict, lr_cfg: Dict):
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 50000),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("min_df", 5),
            max_df=tfidf_cfg.get("max_df", 0.95),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        )
        self.classifier = LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            solver=lr_cfg.get("solver", "lbfgs"),
            multi_class=lr_cfg.get("multi_class", "multinomial"),
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, texts: List[str], labels: np.ndarray):
        logger.info("Fitting TF-IDF + LR baseline...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        self.is_fitted = True
        logger.info(f"TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

    def predict(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def evaluate(self, texts: List[str], labels: np.ndarray) -> Dict:
        preds = self.predict(texts)
        proba = self.predict_proba(texts)
        return compute_metrics(labels, preds, proba)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"vectorizer": self.vectorizer,
                          "classifier": self.classifier}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectorizer = data["vectorizer"]
        self.classifier = data["classifier"]
        self.is_fitted = True


# ============================================================
# Metadata-Only Baselines (replicating the paper's ML models)
# ============================================================

class MetadataBaseline:
    """
    Metadata-only classifiers using the 7 features from the paper.
    Supports multiple algorithms matching the paper's comparison.
    """

    ALGORITHMS = {
        "lr": lambda: LogisticRegression(
            max_iter=1000, class_weight="balanced", multi_class="multinomial"
        ),
        "rf": lambda: RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
        "gbdt": lambda: GradientBoostingClassifier(
            n_estimators=200, random_state=42
        ),
        "knn": lambda: KNeighborsClassifier(n_neighbors=5),
        "svm": lambda: SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=42
        ),
        "mlp": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=500, random_state=42
        ),
        "gnb": lambda: GaussianNB(),
    }

    def __init__(self, algorithm: str = "gbdt",
                 feature_cols: Optional[List[str]] = None):
        self.algorithm = algorithm
        self.feature_cols = feature_cols or [
            "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
            "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
        ]
        self.scaler = StandardScaler()

        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                             f"Available: {list(self.ALGORITHMS.keys())}")
        self.classifier = self.ALGORITHMS[algorithm]()

    def fit(self, df: pd.DataFrame, label_col: str = "cluster"):
        logger.info(f"Fitting metadata baseline ({self.algorithm})...")
        existing = [c for c in self.feature_cols if c in df.columns]
        X = df[existing].values.astype(np.float32)
        X = np.nan_to_num(X)
        X = self.scaler.fit_transform(X)
        y = df[label_col].values
        self.classifier.fit(X, y)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        existing = [c for c in self.feature_cols if c in df.columns]
        X = df[existing].values.astype(np.float32)
        X = np.nan_to_num(X)
        X = self.scaler.transform(X)
        return self.classifier.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        existing = [c for c in self.feature_cols if c in df.columns]
        X = df[existing].values.astype(np.float32)
        X = np.nan_to_num(X)
        X = self.scaler.transform(X)
        if hasattr(self.classifier, "predict_proba"):
            return self.classifier.predict_proba(X)
        # Fallback for classifiers without predict_proba
        return np.eye(4)[self.classifier.predict(X)]

    def evaluate(self, df: pd.DataFrame, label_col: str = "cluster") -> Dict:
        preds = self.predict(df)
        proba = self.predict_proba(df)
        labels = df[label_col].values
        return compute_metrics(labels, preds, proba)


# ============================================================
# Combined TF-IDF + Metadata
# ============================================================

class CombinedBaseline:
    """TF-IDF text features + numerical metadata combined."""

    def __init__(self, tfidf_cfg: Dict, lr_cfg: Dict,
                 feature_cols: Optional[List[str]] = None):
        self.tfidf = TfidfVectorizer(
            max_features=tfidf_cfg.get("max_features", 50000),
            ngram_range=tuple(tfidf_cfg.get("ngram_range", [1, 2])),
            min_df=tfidf_cfg.get("min_df", 5),
            max_df=tfidf_cfg.get("max_df", 0.95),
            sublinear_tf=tfidf_cfg.get("sublinear_tf", True),
        )
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=lr_cfg.get("C", 1.0),
            max_iter=lr_cfg.get("max_iter", 1000),
            class_weight=lr_cfg.get("class_weight", "balanced"),
            multi_class="multinomial",
            random_state=42,
        )
        self.feature_cols = feature_cols or [
            "ACCESS_SIZE", "ACCESS_TREND", "SIM_ACCESS", "SIM_TECH",
            "INVENT_DIVER", "INVENT_APPL", "ATTENT_SIZE",
        ]

    def _build_features(self, texts: List[str], df: pd.DataFrame,
                         fit: bool = False):
        if fit:
            X_text = self.tfidf.fit_transform(texts)
        else:
            X_text = self.tfidf.transform(texts)

        existing = [c for c in self.feature_cols if c in df.columns]
        X_meta = df[existing].values.astype(np.float32)
        X_meta = np.nan_to_num(X_meta)
        if fit:
            X_meta = self.scaler.fit_transform(X_meta)
        else:
            X_meta = self.scaler.transform(X_meta)

        return hstack([X_text, csr_matrix(X_meta)])

    def fit(self, texts: List[str], df: pd.DataFrame, label_col: str = "cluster"):
        X = self._build_features(texts, df, fit=True)
        y = df[label_col].values
        self.classifier.fit(X, y)

    def predict(self, texts: List[str], df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(texts, df, fit=False)
        return self.classifier.predict(X)

    def predict_proba(self, texts: List[str], df: pd.DataFrame) -> np.ndarray:
        X = self._build_features(texts, df, fit=False)
        return self.classifier.predict_proba(X)

    def evaluate(self, texts: List[str], df: pd.DataFrame,
                 label_col: str = "cluster") -> Dict:
        preds = self.predict(texts, df)
        proba = self.predict_proba(texts, df)
        labels = df[label_col].values
        return compute_metrics(labels, preds, proba)


# ============================================================
# Evaluation Utilities
# ============================================================

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_proba: Optional[np.ndarray] = None) -> Dict:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "macro_precision": f1_score(y_true, y_pred, average="macro"),
        "macro_recall": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True
        ),
    }

    if y_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            )
        except ValueError:
            metrics["roc_auc"] = 0.0
        try:
            metrics["cross_entropy"] = log_loss(y_true, y_proba)
        except ValueError:
            metrics["cross_entropy"] = float("inf")

    return metrics
