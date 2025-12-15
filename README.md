## Liquidity Manipulation Detector

This is a Streamlit-based decision-support tool designed for credit analysts and rating agencies.

### Purpose
To detect liquidity manipulation and maturity mismatch using:
- CR vs TACR drift
- DSO escalation
- CFO vs PAT divergence
- AI-based anomaly detection

### How it works
1. Upload Excel file
2. Tool computes financial ratios
3. Flags risky behavior
4. Produces a Liquidity Risk Score and classification
5. Allows download of results

### Stakeholder
Credit Analysts / Rating Agencies for lending validation and early warning.

### AI Usage
Unsupervised Isolation Forest is used to detect abnormal financial patterns.

### Output
Dashboard + downloadable Excel + risk buckets.
