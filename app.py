import streamlit as st
import pandas as pd

st.set_page_config(page_title="ML Model Builder Dashboard", layout="wide")

st.title("ML Model Builder Dashboard")
st.markdown("""
This dashboard helps you understand how ML models are built from a CSV and 
how configuration choices affect performance metrics.
""")

# Placeholder for dataset selection
dataset = st.sidebar.selectbox("Select Dataset", [
    "House Price (Regression)",
    "Titanic (Binary Classification)",
    "MNIST (Multiclass Classification)",
    "Mall Customer (Clustering)"
])

st.write(f"### Current Task: {dataset}")

# Placeholder for model knobs
st.sidebar.header("Model Configuration")
scaling = st.sidebar.selectbox("Scaling", ["Standard", "MinMax", "Robust"])
imputation = st.sidebar.selectbox("Imputation", ["Mean", "Median", "KNN", "MICE"])

st.write("Results and comparisons will appear here based on your configurations.")
