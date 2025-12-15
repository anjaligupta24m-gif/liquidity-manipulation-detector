import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from io import BytesIO

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Liquidity Manipulation Detector",
    layout="wide"
)

st.title("Liquidity Manipulation Detector")
st.caption("AI-based early warning tool for Credit Analysts & Rating Agencies")

# -------------------------------------------------
# File upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Excel file (Financial Statement Data)",
    type=["xlsx"]
)

# -------------------------------------------------
# User-controlled thresholds (Sensitivity)
# -------------------------------------------------
st.sidebar.header("Model Parameters")

dso_threshold = st.sidebar.number_input(
    "DSO Threshold (days)",
    value=120,
    min_value=30,
    max_value=365
)

cfo_pat_threshold = st.sidebar.number_input(
    "CFO / PAT Threshold",
    value=0.7,
    min_value=-5.0,
    max_value=5.0,
    step=0.05
)

weight_option = st.sidebar.selectbox(
    "Risk Scoring Weights",
    [
        "AI 60% - Flags 40%",
        "AI 50% - Flags 50%",
        "AI 70% - Flags 30%"
    ]
)

# -------------------------------------------------
# Main logic
# -------------------------------------------------
if uploaded_file is not None:

    # Read data
    df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # -------------------------------------------------
    # Required columns check
    # -------------------------------------------------
    required_cols = [
        "Company", "Year", "Receivables", "Cash",
        "TradePayables", "ShortTermBorrowings",
        "TotalCurrentAssets", "TotalCurrentLiabilities",
        "CFO", "Sales", "NetProfit"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    # -------------------------------------------------
    # Ratio computation
    # -------------------------------------------------
    df["CR"] = df["TotalCurrentAssets"] / df["TotalCurrentLiabilities"]

    df["TACR"] = (
        df["Cash"] + df["Receivables"]
    ) / (
        df["TradePayables"] + df["ShortTermBorrowings"]
    )

    df["DSO"] = (df["Receivables"] / df["Sales"]) * 365

    # CFO / PAT only meaningful when PAT > 0
    df["CFO_to_PAT"] = np.where(
        df["NetProfit"] > 0,
        df["CFO"] / df["NetProfit"],
        np.nan
    )

    df["CFO_margin"] = df["CFO"] / df["Sales"]

    # -------------------------------------------------
    # Rule-based red flags
    # -------------------------------------------------
    df["Flag_TACR_Mismatch"] = np.where(
        (df["CR"] > 1) & (df["TACR"] < 1), 1, 0
    )

    df["Flag_High_DSO"] = np.where(
        df["DSO"] > dso_threshold, 1, 0
    )

    df["Flag_Low_Cash_Profit"] = np.where(
        df["CFO_to_PAT"] < cfo_pat_threshold, 1, 0
    )

    df["Flag_Count"] = (
        df["Flag_TACR_Mismatch"]
        + df["Flag_High_DSO"]
        + df["Flag_Low_Cash_Profit"]
    )

    # -------------------------------------------------
    # AI anomaly detection (Isolation Forest)
    # -------------------------------------------------
    features = df[["CR", "TACR", "DSO", "CFO_to_PAT"]].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.25,
        random_state=42
    )

    df["AI_raw"] = -model.fit_predict(features)

    # Normalize AI score to 0–100
    df["AI_Anomaly_Score"] = (
        (df["AI_raw"] - df["AI_raw"].min())
        / (df["AI_raw"].max() - df["AI_raw"].min())
    ) * 100

    # -------------------------------------------------
    # Composite Liquidity Risk Score
    # -------------------------------------------------
    df["Flag_Score"] = df["Flag_Count"] * 33  # max ~100

    if weight_option == "AI 60% - Flags 40%":
        df["Liquidity_Risk_Score"] = (
            0.6 * df["AI_Anomaly_Score"] + 0.4 * df["Flag_Score"]
        )
    elif weight_option == "AI 50% - Flags 50%":
        df["Liquidity_Risk_Score"] = (
            0.5 * df["AI_Anomaly_Score"] + 0.5 * df["Flag_Score"]
        )
    else:
        df["Liquidity_Risk_Score"] = (
            0.7 * df["AI_Anomaly_Score"] + 0.3 * df["Flag_Score"]
        )

    # -------------------------------------------------
    # Risk Buckets
    # -------------------------------------------------
    def risk_bucket(x):
        if x >= 70:
            return "High"
        elif x >= 40:
            return "Medium"
        else:
            return "Low"

    df["Risk_Bucket"] = df["Liquidity_Risk_Score"].apply(risk_bucket)

    # -------------------------------------------------
    # Dashboard outputs
    # -------------------------------------------------
    st.subheader("Portfolio-Level Insights")

    col1, col2, col3 = st.columns(3)
    col1.metric("Companies", df["Company"].nunique())
    col2.metric("High-Risk Firm-Years", (df["Risk_Bucket"] == "High").sum())
    col3.metric("Avg Liquidity Risk Score", round(df["Liquidity_Risk_Score"].mean(), 2))

    st.subheader("Average Liquidity Risk Over Time")
    st.line_chart(df.groupby("Year")["Liquidity_Risk_Score"].mean())

    st.subheader("Firm-Year Risk Assessment")
    st.dataframe(
        df.sort_values("Liquidity_Risk_Score", ascending=False),
        use_container_width=True
    )

    # -------------------------------------------------
    # Download results (FIXED)
    # -------------------------------------------------
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)

    st.download_button(
        label="Download Results (Excel)",
        data=output,
        file_name="liquidity_risk_output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.info("👆 Upload an Excel file to begin analysis.")
