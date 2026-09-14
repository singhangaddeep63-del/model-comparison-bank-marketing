"""
Train and compare multiple classification models on the Bank Marketing
dataset, one by one, on identical train/test splits and preprocessing.

Models covered:
  - Logistic Regression        (linear baseline)
  - K-Nearest Neighbors        (instance-based)
  - Decision Tree              (single tree baseline)
  - Random Forest              (bagging)
  - AdaBoost                   (boosting, classic)
  - Gradient Boosting          (boosting, sklearn native)
  - XGBoost                    (boosting, industry standard)
  - Stacking Classifier        (ensemble of the above)
"""
import json
import time
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, StackingClassifier,
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from xgboost import XGBClassifier

from data_prep import load_data, build_preprocessor, get_train_test_split

RANDOM_STATE = 42

MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "knn": KNeighborsClassifier(n_neighbors=15),
    "decision_tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "adaboost": AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", random_state=RANDOM_STATE,
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
    fitted_pipelines = {}
    best_name, best_score, best_pipeline = None, -1, None

    for name, clf in MODELS.items():
        start = time.time()
        pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)
        elapsed = round(time.time() - start, 1)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = evaluate(y_test, y_pred, y_proba)
        metrics["train_seconds"] = elapsed
        results[name] = metrics
        fitted_pipelines[name] = pipe
        print(f"{name}: {metrics}")

        if metrics["roc_auc"] > best_score:
            best_name, best_score, best_pipeline = name, metrics["roc_auc"], pipe

    # Stacking ensemble built from the three strongest individual models
    stack_estimators = [
        ("rf", fitted_pipelines["random_forest"].named_steps["clf"]),
        ("gb", fitted_pipelines["gradient_boosting"].named_steps["clf"]),
        ("xgb", fitted_pipelines["xgboost"].named_steps["clf"]),
    ]
    stack = StackingClassifier(
        estimators=stack_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3,
    )
    stack_pipe = Pipeline(steps=[("prep", build_preprocessor()), ("clf", stack)])
    start = time.time()
    stack_pipe.fit(X_train, y_train)
    elapsed = round(time.time() - start, 1)
    y_pred = stack_pipe.predict(X_test)
    y_proba = stack_pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, y_pred, y_proba)
    metrics["train_seconds"] = elapsed
    results["stacking_ensemble"] = metrics
    print(f"stacking_ensemble: {metrics}")

    if metrics["roc_auc"] > best_score:
        best_name, best_score, best_pipeline = "stacking_ensemble", metrics["roc_auc"], stack_pipe

    joblib.dump(best_pipeline, "models/best_model.joblib")
    joblib.dump(fitted_pipelines["xgboost"], "models/xgboost_model.joblib")  # for SHAP script
    with open("models/metrics.json", "w") as f:
        json.dump({"results": results, "best_model": best_name}, f, indent=2)

    print(f"\nBest model: {best_name} (ROC-AUC={best_score})")
    print("Saved to models/best_model.joblib")


if __name__ == "__main__":
    main()
