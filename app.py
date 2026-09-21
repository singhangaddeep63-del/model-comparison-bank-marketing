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

st.subheader("Why the model predicts what it predicts")
st.markdown(
    "SHAP values for the XGBoost model — each dot is one customer, color shows "
    "whether that feature's value was high or low, and position shows whether "
    "it pushed the prediction toward churn (right) or not (left)."
)
st.image("models/shap_summary.png", use_container_width=True)
