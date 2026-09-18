"""Generate SHAP explanations for the trained XGBoost model."""
import joblib
import matplotlib.pyplot as plt
import shap

from data_prep import load_data, get_train_test_split

def main():
    df = load_data()
    X_train, X_test, y_train, y_test = get_train_test_split(df)

    pipe = joblib.load("models/xgboost_model.joblib")
    preprocessor = pipe.named_steps["prep"]
    model = pipe.named_steps["clf"]

    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    plt.figure()
    shap.summary_plot(
        shap_values, X_test_transformed, feature_names=feature_names, show=False
    )
    plt.tight_layout()
    plt.savefig("models/shap_summary.png", dpi=150)
    print("Saved models/shap_summary.png")


if __name__ == "__main__":
    main()
