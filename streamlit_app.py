import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Liquidity Manipulation Detector", layout="wide")

st.title("Liquidity Manipulation Detector — Upload & Analyze")
st.caption("Credit Analyst Tool | Upload Excel → Automated Liquidity Risk Assessment")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

# Threshold inputs
dso_threshold = st.number_input("DSO threshold (days)", value=120)
cfo_pat_threshold = st.number_input("CFO / PAT threshold", value=0.7)

weight_option = st.selectbox(
    "Weight scenario",
    ["AI 60% - Flags 40%", "AI 50% - Flags 50%", "AI 70% - Flags 30%"]
)

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # ---- Core Ratios ----
    df["CR"] = df["TotalCurrentAssets"] / df["TotalCurrentLiabilities"]
    df["TACR"] = (df["Cash"] + df["Receivables"]) / (df["TradePayables"] + df["ShortTermBorrowings"])
    df["DSO"] = (df["Receivables"] / df["Sales"]) * 365
    df["CFO_to_PAT"] = df["CFO"] / df["NetProfit"].replace(0, np.nan)

    # ---- Flags ----
    df["flag_DSO"] = (df["DSO"] > dso_threshold).astype(int)
    df["flag_TACR"] = ((df["CR"] > 1) & (df["TACR"] < 1)).astype(int)
    df["flag_CFO"] = (df["CFO_to_PAT"] < cfo_pat_threshold).astype(int)

    df["Flag_Count"] = df[["flag_DSO", "flag_TACR", "flag_CFO"]].sum(axis=1)

    # ---- AI Anomaly Detection ----
    model = IsolationForest(contamination=0.25, random_state=42)
    features = df[["CR", "TACR", "DSO", "CFO_to_PAT"]].fillna(0)
    df["AI_Anomaly_Score"] = -model.fit_predict(features)
    df["AI_Anomaly_Score"] = (df["AI_Anomaly_Score"] - df["AI_Anomaly_Score"].min()) / (
        df["AI_Anomaly_Score"].max() - df["AI_Anomaly_Score"].min()
    ) * 100

    # ---- Weighting ----
    if weight_option == "AI 60% - Flags 40%":
        df["Liquidity_Risk_Score"] = 0.6 * df["AI_Anomaly_Score"] + 0.4 * (df["Flag_Count"] * 33)
    elif weight_option == "AI 50% - Flags 50%":
        df["Liquidity_Risk_Score"] = 0.5 * df["AI_Anomaly_Score"] + 0.5 * (df["Flag_Count"] * 33)
    else:
        df["Liquidity_Risk_Score"] = 0.7 * df["AI_Anomaly_Score"] + 0.3 * (df["Flag_Count"] * 33)

    # ---- Risk Buckets ----
    def bucket(x):
        if x > 70:
            return "High"
        elif x > 40:
            return "Medium"
        else:
            return "Low"

    df["Risk_Bucket"] = df["Liquidity_Risk_Score"].apply(bucket)

    # ---- Dashboard ----
    st.subheader("Summary")
    st.write(f"Companies analyzed: {df['Company'].nunique()}")
    st.write(f"High-risk firm-years: {(df['Risk_Bucket']=='High').sum()}")

    st.subheader("Liquidity Risk Scores")
    st.line_chart(df.groupby("Year")["Liquidity_Risk_Score"].mean())

    st.subheader("Results Table")
    st.dataframe(df)

    # ---- Download ----
    st.download_button(
        "Download Results (Excel)",
        data=df.to_excel(index=False),
        file_name="liquidity_risk_output.xlsx"
    )
