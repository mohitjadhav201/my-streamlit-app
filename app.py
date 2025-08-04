import os
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Data Validation Tool", layout="wide")
st.title("🧪 Data Validation Tool")

uploaded_file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Preview of Data")
    st.dataframe(df.head())

    if st.button("Run Validation Checks"):
        st.header("🧹 Validation Results")

        # Dimensions & Data Types
        st.subheader("📏 Dataset Dimensions and Types")
        st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
        st.write("**Data Types:**")
        st.dataframe(df.dtypes)

        # Missing Values
        st.subheader("❌ Missing Values")
        st.dataframe(df.isnull().sum().reset_index().rename(columns={0: "Missing Count"}))

        # Duplicate Rows
        st.subheader("📎 Duplicate Rows")
        st.write(f"**Duplicate Rows:** {df.duplicated().sum()}")

        # Column Uniqueness
        st.subheader("🔑 Unique Value Analysis")
        unique_counts = df.nunique().reset_index()
        unique_counts.columns = ["Column", "Unique Values"]
        st.dataframe(unique_counts)

        # Categorical Summary
        st.subheader("📦 Categorical Column Summary")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                st.write(f"**{col}** – Top 5 most frequent:")
                st.write(df[col].value_counts().head())
        else:
            st.write("No categorical columns detected.")

        # Outlier Detection
        st.subheader("🚨 Outlier Detection (IQR method)")
        num_cols = df.select_dtypes(include=['number']).columns
        outlier_summary = []
        for col in num_cols:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
            outlier_summary.append({"Column": col, "Outliers": outliers})
        st.dataframe(pd.DataFrame(outlier_summary))

        # Correlation Heatmap
        if len(num_cols) >= 2:
            st.subheader("🧪 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough numeric columns for correlation heatmap.")

        # Distribution Plots
        st.subheader("📉 Distribution of Numeric Features")
        for col in num_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

