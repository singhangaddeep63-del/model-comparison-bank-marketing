"""Streamlit dashboard: model comparison results + SHAP explanations."""
import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Model Comparison — Bank Marketing", page_icon="📊", layout="wide")
st.title("Model Comparison: Bank Term Deposit Prediction")
st.caption("45,211 customers · Portuguese bank telemarketing campaign · UCI Bank Marketing dataset")

with open("models/metrics.json") as f:
    data = json.load(f)

results_df = pd.DataFrame(data["results"]).T
results_df.index.name = "model"
results_df = results_df.sort_values("roc_auc", ascending=False)

st.subheader("Results — all models, same split & preprocessing")
st.dataframe(
    results_df.style.highlight_max(subset=["accuracy", "precision", "recall", "f1", "roc_auc"], color="#1a4d2e"),
    use_container_width=True,
)
st.caption(f"Best model by ROC-AUC: **{data['best_model']}**")

st.bar_chart(results_df["roc_auc"])

st.subheader("SMOTE vs class_weight — does oversampling help?")
try:
    with open("models/smote_comparison.json") as f:
        smote_data = json.load(f)
    smote_df = pd.DataFrame(smote_data).T
    smote_df.index.name = "model_approach"
    st.dataframe(smote_df, use_container_width=True)
    st.caption(
        "SMOTE applied only to training data. Across these models, SMOTE "
        "doesn't clearly beat class_weight='balanced' on ROC-AUC — it mainly "
        "trades recall for precision instead."
    )
except FileNotFoundError:
    st.info("Run `python src/smote_comparison.py` to generate this comparison.")

st.subheader("Why the model predicts what it predicts")
st.markdown(
    "SHAP values for the XGBoost model — each dot is one customer, color shows "
    "whether that feature's value was high or low, and position shows whether "
    "it pushed the prediction toward churn (right) or not (left)."
)
st.image("models/shap_summary.png", use_container_width=True)
