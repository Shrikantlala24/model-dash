import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    page_title="Model-Dash | Educational ML Pipeline",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load config and experiment results
@st.cache_resource
def load_config():
    """Load dataset configuration from YAML"""
    config_path = Path("config/datasets.yaml")
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f)
    return None

@st.cache_resource
def load_experiments():
    """Load precomputed experiment results"""
    results_path = Path("experiments/results_db.json")
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}

# Load data
config = load_config()
experiments = load_experiments()

# Educational explanations for each preprocessing stage
EXPLANATIONS = {
    "imputation": {
        "Mean": "🔢 Simple average. Fast but **biased if data is skewed**. Best for normally distributed data.",
        "Median": "📊 Resistant to outliers. Better choice for **skewed distributions** with extreme values.",
        "KNN": "👥 Finds similar rows and uses their values. **Preserves relationships**—slower but more accurate.",
        "MICE": "🧬 Iteratively models relationships. **Gold standard** but computationally expensive. Best for complex data."
    },
    "scaling": {
        "None": "✋ No scaling. Use only with tree-based models (Random Forest, XGBoost).",
        "Standard": "📏 Z-score normalization (mean=0, std=1). **Best for linear models & neural networks**. Assumes normal distribution.",
        "MinMax": "📈 Scales to [0, 1]. **Good for neural networks**. ⚠️ Sensitive to outliers.",
        "Robust": "🛡️ Uses median & IQR—**robust to outliers**. Best choice when outliers present."
    },
    "encoding": {
        "One-Hot": "📍 Binary columns per category. **Standard choice**—treats categories as independent.",
        "Ordinal": "🔢 Maps to integers (assumes order). **Compact** but risky if no natural order exists.",
        "Target": "🎯 Encodes with mean target value. **Powerful** but risk of overfitting on small samples."
    },
    "outlier_handling": {
        "Remove": "🗑️ Delete outlier rows. Loses information but clean data.",
        "Cap": "🔗 Replace with 95th/5th percentile. Keeps information while reducing extreme influence.",
        "Transform": "🔄 Log/sqrt transform. Makes distributions normal—helps linear models.",
        "Keep": "✅ Accept outliers as legitimate signals. Tree-based models handle this well."
    }
}

# Problem-type specific metadata
PROBLEM_METADATA = {
    "regression": {
        "display_name": "🔮 Regression",
        "description": "Predict continuous numerical values (e.g., house prices)",
        "target_type": "continuous",
        "metrics": ["R²", "RMSE", "MAE"],
        "model_options": ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"],
        "key_insight": "**Why it matters:** Different scaling methods directly impact gradient-descent models. Distance-based features matter enormously."
    },
    "binary_classification": {
        "display_name": "✅❌ Binary Classification",
        "description": "Predict one of two classes (e.g., Titanic survival)",
        "target_type": "binary",
        "metrics": ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"],
        "model_options": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"],
        "key_insight": "**Why it matters:** Class imbalance changes which metric to trust. Accuracy can lie if one class dominates."
    },
    "multiclass_classification": {
        "display_name": "🏷️ Multiclass Classification",
        "description": "Predict one of multiple classes (e.g., Fashion MNIST - 10 classes)",
        "target_type": "multiclass",
        "metrics": ["Accuracy", "Macro F1", "Weighted F1", "Per-Class Recall"],
        "model_options": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Neural Network"],
        "key_insight": "**Why it matters:** Categorical encoding (One-Hot vs. Ordinal) becomes critical. Per-class metrics reveal imbalance."
    },
    "clustering": {
        "display_name": "👥 Clustering",
        "description": "Group similar items unsupervised (e.g., customer segmentation)",
        "target_type": "unsupervised",
        "metrics": ["Silhouette Score", "Davies-Bouldin", "Inertia", "Calinski-Harabasz"],
        "model_options": ["K-Means", "DBSCAN", "Hierarchical", "Gaussian Mixture"],
        "key_insight": "**Why it matters:** Scaling is **critical** for K-Means (Euclidean distance). Unscaled features dominate."
    },
    "anomaly_detection": {
        "display_name": "🚨 Anomaly Detection",
        "description": "Identify rare/unusual items (e.g., credit card fraud)",
        "target_type": "imbalanced_binary",
        "metrics": ["Precision", "Recall", "F1 (Minority)", "AUC-ROC"],
        "model_options": ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "Neural Network"],
        "key_insight": "**Why it matters:** Accuracy is **misleading**. Focus on precision/recall of the minority class (fraud)."
    },
    "time_series": {
        "display_name": "📈 Time Series",
        "description": "Predict future values from temporal patterns (e.g., stock returns)",
        "target_type": "temporal",
        "metrics": ["R²", "RMSE", "MAE", "Directional Accuracy"],
        "model_options": ["ARIMA", "Prophet", "LSTM", "XGBoost", "LightGBM"],
        "key_insight": "**Why it matters:** Lagged features & moving averages are essential. Train-test split must respect time order."
    }
}

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================

st.sidebar.title("🛠️ Pipeline Controls")
st.sidebar.markdown("*Configure your ML experiment. Metrics are precomputed for speed.*")
st.sidebar.markdown("---")

# Problem type selector
st.sidebar.subheader("1️⃣ Problem Type")
problem_type = st.sidebar.selectbox(
    "Choose the ML task:",
    options=list(PROBLEM_METADATA.keys()),
    format_func=lambda x: PROBLEM_METADATA[x]["display_name"]
)

problem_meta = PROBLEM_METADATA[problem_type]

# Dataset selector (filtered by problem type)
st.sidebar.subheader("2️⃣ Dataset")
available_datasets = [
    d for d, meta in (config.get("datasets", {}) or {}).items()
    if meta.get("problem_type") == problem_type
] if config else []

dataset_key = st.sidebar.selectbox(
    "Select dataset:",
    options=available_datasets,
    key="dataset_select"
) if available_datasets else None

if dataset_key and config:
    dataset_meta = config["datasets"][dataset_key]
    st.sidebar.info(f"📊 {dataset_meta.get('description', 'No description')}")

st.sidebar.markdown("---")

# Problem-specific preprocessing options
st.sidebar.subheader("3️⃣ Preprocessing Pipeline")

# Imputation (skip for unsupervised)
if problem_type != "clustering":
    imputation = st.sidebar.select_slider(
        "Missing Value Handling",
        options=["Mean", "Median", "KNN", "MICE"],
        value="Median"
    )
    st.sidebar.caption(EXPLANATIONS["imputation"].get(imputation, ""))
else:
    imputation = None

# Scaling (always relevant except clustering might use different approaches)
scaling = st.sidebar.select_slider(
    "Feature Scaling",
    options=["None", "Standard", "MinMax", "Robust"],
    value="Standard"
)
st.sidebar.caption(EXPLANATIONS["scaling"].get(scaling, ""))

# Encoding (for classification only)
if "classification" in problem_type or problem_type == "anomaly_detection":
    encoding = st.sidebar.selectbox(
        "Categorical Encoding",
        options=["One-Hot", "Ordinal", "Target"],
        key="encoding"
    )
    st.sidebar.caption(EXPLANATIONS["encoding"].get(encoding, ""))
else:
    encoding = None

st.sidebar.markdown("---")

# Model selection
st.sidebar.subheader("4️⃣ Model Selection")
model = st.sidebar.selectbox(
    "Algorithm:",
    options=problem_meta["model_options"]
)

st.sidebar.markdown("---")

# Experiment trigger
st.sidebar.info("✅ All results are precomputed. No model training happens here—just visualization.")

# ============================================================================
# MAIN AREA - EDUCATIONAL DASHBOARD
# ============================================================================

st.title("🎓 Model-Dash: ML Pipeline Exploration")

# Header with selected configuration
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.metric("Problem Type", problem_meta["display_name"].split(" ", 1)[1])
with col2:
    if dataset_key and config:
        st.markdown(f"**Dataset:** `{dataset_key}`")
with col3:
    st.metric("Model", model.split(" ")[0])

st.markdown("---")

# Section 1: WHY THIS MATTERS (Educational Context)
st.subheader("💡 Why Your Choices Matter")
st.info(problem_meta["key_insight"])

# Section 2: Configuration Breakdown
st.subheader("⚙️ Your Pipeline Configuration")
config_cols = st.columns(4 if problem_type != "clustering" else 3)

with config_cols[0]:
    if imputation:
        st.text("Imputation")
        st.code(imputation, language="text")

with config_cols[1]:
    st.text("Scaling")
    st.code(scaling, language="text")

if len(config_cols) > 2 and encoding:
    with config_cols[2]:
        st.text("Encoding")
        st.code(encoding, language="text")

with config_cols[-1]:
    st.text("Model")
    st.code(model, language="text")

st.markdown("---")

# Section 3: Expected Metrics for This Task Type
st.subheader("📊 Relevant Metrics for This Task")
metrics_text = " • ".join(problem_meta["metrics"])
st.markdown(f"**Track:** {metrics_text}")

if problem_type == "regression":
    st.write("""
    - **R² (Coefficient of Determination):** How much variance your model explains. Range: 0-1. Higher is better.
    - **RMSE:** Average magnitude of prediction errors. Lower is better. Same units as target.
    - **MAE:** Mean absolute error. Less sensitive to outliers than RMSE.
    """)
elif "classification" in problem_type:
    st.write("""
    - **Accuracy:** Correct predictions / total predictions. Misleading if classes are imbalanced.
    - **Precision:** Of predicted positives, how many were correct? Focus on false positives.
    - **Recall:** Of actual positives, how many did we catch? Focus on false negatives.
    - **F1:** Harmonic mean of precision & recall. Use when both matter equally.
    - **AUC-ROC:** Measures performance across all probability thresholds. 0.5=random, 1.0=perfect.
    """)

st.markdown("---")

# Section 4: Experiment Results & Comparisons (if data available)
st.subheader("🔬 Experiment Results")

# Mock experiment results (would load from experiments/results_db.json in production)
if dataset_key and experiments:
    if dataset_key in experiments:
        exp_data = experiments[dataset_key]
        results = exp_data.get("results", [])
        
        # Filter results by current config
        matching_results = [
            r for r in results
            if r.get("scaling") == scaling
            and (imputation is None or r.get("imputation") == imputation)
            and r.get("model") == model
        ]
        
        if matching_results:
            best_result = matching_results[0]
            
            # Display key metrics
            metric_cols = st.columns(len(problem_meta["metrics"]))
            for idx, metric in enumerate(problem_meta["metrics"][:len(metric_cols)]):
                with metric_cols[idx]:
                    value = best_result.get("metrics", {}).get(metric, "N/A")
                    st.metric(metric, f"{value:.3f}" if isinstance(value, float) else value)
else:
    st.warning("⏳ Experiment results not yet loaded. Run `scripts/precompute_experiments.py` first.")

st.markdown("---")

# Section 5: Impact Visualization - Compare variations
st.subheader("📈 How Choices Impact Results")

# Create comparison chart: this scaling vs. others with same model
comparison_data = []
if dataset_key and experiments and dataset_key in experiments:
    results = experiments[dataset_key].get("results", [])
    
    # Compare different scaling methods with same imputation & model
    for result in results:
        if (result.get("imputation") == imputation and 
            result.get("model") == model):
            comparison_data.append({
                "Scaling": result.get("scaling", "Unknown"),
                problem_meta["metrics"][0]: result.get("metrics", {}).get(problem_meta["metrics"][0], 0)
            })

if comparison_data:
    df_comparison = pd.DataFrame(comparison_data)
    fig_scaling = px.bar(
        df_comparison,
        x="Scaling",
        y=problem_meta["metrics"][0],
        title=f"Impact of Scaling on {problem_meta['metrics'][0]} (Fixed: {imputation or 'N/A'} imputation, {model})",
        color="Scaling",
        text_auto=".3f"
    )
    fig_scaling.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_scaling, use_container_width=True)
else:
    st.info("📊 Comparison charts will appear once experiments are precomputed.")

st.markdown("---")

# Section 6: Preprocessing Explanation Deep Dive
st.subheader("🔍 Understanding This Stage: What Happens?")

tabs = st.tabs(["Imputation", "Scaling", "Encoding", "Why It Matters"])

with tabs[0]:
    if imputation:
        st.markdown(f"### {imputation} Imputation")
        st.write(EXPLANATIONS["imputation"].get(imputation, ""))
        if imputation == "KNN":
            st.write("**Algorithm:** Find K nearest rows (by Euclidean distance), average their values.")
        elif imputation == "MICE":
            st.write("**Algorithm:** Iteratively model each column with missing values as target, create multiple plausible datasets.")
        elif imputation == "Mean":
            st.write("**Formula:** `x̄ = Σx / n`")
        elif imputation == "Median":
            st.write("**Definition:** The middle value when sorted. Robust to extreme values.")

with tabs[1]:
    st.markdown(f"### {scaling} Scaling")
    st.write(EXPLANATIONS["scaling"].get(scaling, ""))
    if scaling == "Standard":
        st.write("""```
    x_scaled = (x - mean) / std_dev
    Result: mean=0, std=1
        ```""")
    elif scaling == "MinMax":
        st.write("""```
    x_scaled = (x - min) / (max - min)
    Result: range [0, 1]
        ```""")
    elif scaling == "Robust":
        st.write("""```
    x_scaled = (x - median) / IQR
    Result: robust to outliers
        ```""")

with tabs[2]:
    if encoding:
        st.markdown(f"### {encoding} Encoding")
        st.write(EXPLANATIONS["encoding"].get(encoding, ""))
        if encoding == "One-Hot":
            st.write("**Example:** Color=[Red, Blue, Green] → 3 new binary columns")
        elif encoding == "Target":
            st.write("**Example:** For Titanic, Gender → Female:0.74, Male:0.19 (survival rates)")

with tabs[3]:
    st.markdown("### Why Does This Pipeline Order Matter?")
    st.write("""
    The order is **critical:**
    
    1. **Imputation first** → Fill missing values before you can compute statistics
    2. **Outlier handling second** → Remove/cap extremes before scaling
    3. **Scaling third** → Normalize using statistics from cleaned data
    4. **Encoding last** → One-hot creates new binary features—scale these too!
    
    **Why not reverse it?** Scaling before imputation = biased statistics. Encoding before scaling = wrong scale ranges.
    """)

st.markdown("---")

# Footer
st.markdown("""
---
**💭 Reflection:** Which preprocessing choice surprised you most? Change a knob and compare how results shift!
""")