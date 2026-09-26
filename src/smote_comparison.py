"""
Compare SMOTE oversampling against class_weight='balanced' on the same
models, to see which imbalance-handling approach actually helps on this
dataset.

Important: SMOTE is applied ONLY to the training data, inside the pipeline,
so the test set stays untouched (real, unmodified customers) — otherwise
the evaluation would be testing on fake synthetic data, which is invalid.
"""
import json
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

from data_prep import load_data, build_preprocessor, get_train_test_split

RANDOM_STATE = 42

# Models compared both ways. XGBoost uses scale_pos_weight as its own
# built-in imbalance handling (its equivalent of class_weight).
MODELS = {
    "logistic_regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        LogisticRegression(max_iter=1000),
    ),
    "random_forest": (
        RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE),
        RandomForestClassifier(n_estimators=300, max_depth=10, random_state=RANDOM_STATE),
    ),
    "xgboost": (
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                       eval_metric="logloss", random_state=RANDOM_STATE,
                       scale_pos_weight=7.5),  # approx (majority/minority) ratio
        XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                       eval_metric="logloss", random_state=RANDOM_STATE),
    ),
}


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    results = {}

    for name, (weighted_clf, plain_clf) in MODELS.items():
        # Approach 1: class_weight / scale_pos_weight (no synthetic data)
        pipe_weighted = Pipeline(steps=[("prep", build_preprocessor()), ("clf", weighted_clf)])
        pipe_weighted.fit(X_train, y_train)
        y_pred = pipe_weighted.predict(X_test)
        y_proba = pipe_weighted.predict_proba(X_test)[:, 1]
        results[f"{name}_class_weight"] = evaluate(y_test, y_pred, y_proba)

        # Approach 2: SMOTE — oversample minority class in training data only
        pipe_smote = ImbPipeline(steps=[
            ("prep", build_preprocessor()),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", plain_clf),
        ])
        pipe_smote.fit(X_train, y_train)
        y_pred = pipe_smote.predict(X_test)
        y_proba = pipe_smote.predict_proba(X_test)[:, 1]
        results[f"{name}_smote"] = evaluate(y_test, y_pred, y_proba)

        print(f"{name} (class_weight): {results[f'{name}_class_weight']}")
        print(f"{name} (SMOTE):        {results[f'{name}_smote']}")

    with open("models/smote_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved models/smote_comparison.json")


if __name__ == "__main__":
    main()
