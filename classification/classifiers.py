"""
Training and evaluation of classical classifiers on spectral features.

At least two algorithms are provided by default:
    - Random Forest
    - Support Vector Machine (RBF kernel)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_default_classifiers() -> Dict[str, Any]:
    """
    Return a dictionary of ready-to-use sklearn estimators.

    Returns
    -------
    dict
        name → estimator (unfitted).
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "SVM_RBF": SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            random_state=42,
        ),
    }


def train_and_evaluate(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray | pd.Series,
    classifiers: Dict[str, Any] | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """
    Run stratified k-fold cross-validation for each classifier and
    collect standard metrics.

    Parameters
    ----------
    X : array-like
        Feature matrix (n_samples, n_features).
    y : array-like
        Target labels.
    classifiers : dict, optional
        Mapping name → estimator. Defaults to `get_default_classifiers()`.
    n_splits : int
        Number of CV folds.
    random_state : int
        Seed for the CV splitter.

    Returns
    -------
    dict
        Nested dictionary:
        {
            "RandomForest": {
                "accuracy": float,
                "f1_macro": float,
                "report": str,
                "confusion_matrix": np.ndarray,
                "y_true": np.ndarray,
                "y_pred": np.ndarray,
            },
            ...
        }
    """
    if classifiers is None:
        classifiers = get_default_classifiers()

    if isinstance(X, pd.DataFrame):
        X = X.values
    y = np.asarray(y)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results: Dict[str, Dict[str, Any]] = {}

    for name, clf in classifiers.items():
        # Standardise features inside a pipeline so that the
        # transformation is applied correctly inside each fold.
        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        report = classification_report(y, y_pred, digits=3)
        cm = confusion_matrix(y, y_pred)

        results[name] = {
            "accuracy": float(acc),
            "f1_macro": float(f1),
            "report": report,
            "confusion_matrix": cm,
            "y_true": y,
            "y_pred": y_pred,
        }
        print(f"\n=== {name} ===")
        print(f"Accuracy : {acc:.3f}")
        print(f"F1-macro : {f1:.3f}")
        print(report)

    return results
