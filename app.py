import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

st.set_page_config(page_title="ML Model Builder Dashboard", layout="wide")

st.title("ML Model Builder Dashboard")
st.markdown("Learn how ML configuration affects model performance.")

# Dataset Loading
@st.cache_data
def load_data(name):
    if name == "House Price (Regression)":
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        return df, "MedHouseVal"
    return pd.DataFrame(), None

# Sidebar Configuration
st.sidebar.header("Configuration")
dataset_name = st.sidebar.selectbox("Dataset", ["House Price (Regression)"])
scaling_type = st.sidebar.selectbox("Scaling", ["Standard", "MinMax", "Robust"])
imputation_type = st.sidebar.selectbox("Imputation", ["Mean", "Median", "KNN"])

# Load Data
df, target_col = load_data(dataset_name)

if not df.empty:
    st.write(f"### Current Dataset: {dataset_name}")
    st.dataframe(df.head())

    # Preprocessing Logic (simplified)
    st.sidebar.subheader("Run Experiment")
    if st.sidebar.button("Train Model"):
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Imputation
        if imputation_type == "Mean":
            imputer = SimpleImputer(strategy='mean')
        elif imputation_type == "Median":
            imputer = SimpleImputer(strategy='median')
        else:
            imputer = KNNImputer(n_neighbors=5)
        
        X_imputed = imputer.fit_transform(X)
        
        # Scaling
        if scaling_type == "Standard":
            scaler = StandardScaler()
        elif scaling_type == "MinMax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        X_scaled = scaler.fit_transform(X_imputed)
        
        st.success("Preprocessing complete! Data ready for training.")
        st.write("Processed data shape:", X_scaled.shape)
else:
    st.warning("Please select a valid dataset.")
