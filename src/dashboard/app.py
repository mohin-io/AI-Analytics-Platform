"""
Streamlit Dashboard for the Unified AI Analytics Platform

Interactive web interface for model training, evaluation, and comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing import DataLoader, FeatureEngineer, MissingValueHandler
from src.models.supervised import (
    XGBoostClassifierModel,
    RandomForestClassifierModel,
    LogisticRegressionModel,
    XGBoostRegressorModel,
    RandomForestRegressorModel,
    LinearRegressionModel
)
from src.evaluation import Evaluator, ModelComparator
from src.automl import AutoMLOptimizer

# Page configuration
st.set_page_config(
    page_title="AI Analytics Platform",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}

# Title
st.markdown('<div class="main-header">AI Analytics Platform</div>', unsafe_allow_html=True)
st.markdown("### Automated Machine Learning Model Training and Evaluation")

# Sidebar
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Select Page",
        ["Data Upload", "Model Training", "Model Comparison", "AutoML", "About"]
    )

# Page: Data Upload
if page == "Data Upload":
    st.markdown('<div class="sub-header">📁 Data Upload</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=['csv'])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.data = df

            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")

            # Display data info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())

            # Preview data
            st.subheader("Data Preview")
            st.dataframe(df.head(10))

            # Column info
            st.subheader("Column Information")
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing': df.isnull().sum(),
                'Unique': df.nunique()
            })
            st.dataframe(col_info)

        except Exception as e:
            st.error(f"Error loading dataset: {e}")

# Page: Model Training
elif page == "Model Training":
    st.markdown('<div class="sub-header">🎯 Model Training</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.data

        col1, col2 = st.columns(2)

        with col1:
            task_type = st.selectbox("Task Type", ["classification", "regression"])

        with col2:
            target_column = st.selectbox("Target Column", df.columns)

        algorithm = st.selectbox(
            "Select Algorithm",
            ["XGBoost", "Random Forest", "Logistic Regression", "Linear Regression"]
        )

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)

        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Prepare data
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]

                    # Preprocess
                    engineer = FeatureEngineer(scaling='standard', encoding='onehot')
                    X_processed = engineer.fit_transform(X, y)

                    # Split
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y, test_size=test_size, random_state=42
                    )

                    # Select model
                    if algorithm == "XGBoost":
                        model = XGBoostClassifierModel() if task_type == "classification" else XGBoostRegressorModel()
                    elif algorithm == "Random Forest":
                        model = RandomForestClassifierModel() if task_type == "classification" else RandomForestRegressorModel()
                    elif algorithm == "Logistic Regression":
                        model = LogisticRegressionModel()
                    else:
                        model = LinearRegressionModel()

                    # Train
                    model.train(X_train, y_train)

                    # Evaluate
                    metrics = model.evaluate(X_test, y_test)

                    # Store
                    model_name = f"{algorithm}_{len(st.session_state.models) + 1}"
                    st.session_state.models[model_name] = model
                    st.session_state.results[model_name] = metrics

                    st.success(f"✅ Model trained successfully in {model.training_time:.2f}s!")

                    # Display metrics
                    st.subheader("Model Performance")
                    metric_df = pd.DataFrame([metrics]).T
                    metric_df.columns = ['Value']
                    st.dataframe(metric_df)

                    # Visualizations
                    if task_type == "classification":
                        evaluator = Evaluator(task='classification')
                        predictions = model.predict(X_test)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.pyplot(evaluator.plot_confusion_matrix(y_test, predictions))
                        with col2:
                            if hasattr(model, 'predict_proba'):
                                probs = model.predict_proba(X_test)
                                st.pyplot(evaluator.plot_roc_curve(y_test, probs[:, 1]))

                except Exception as e:
                    st.error(f"Error: {e}")

# Page: Model Comparison
elif page == "Model Comparison":
    st.markdown('<div class="sub-header">📊 Model Comparison</div>', unsafe_allow_html=True)

    if not st.session_state.results:
        st.warning("⚠️ Train some models first!")
    else:
        comparator = ModelComparator()
        comparison_df = comparator.create_comparison_table(st.session_state.results)

        st.subheader("Model Performance Comparison")
        st.dataframe(comparison_df)

        # Leaderboard
        st.subheader("Leaderboard")
        metric_to_rank = st.selectbox("Rank by Metric", comparison_df.columns)
        leaderboard = comparator.generate_leaderboard(comparison_df, metric_to_rank)
        st.dataframe(leaderboard)

        # Best model
        best_model = comparator.get_best_model(comparison_df, metric_to_rank)
        st.success(f"🏆 Best Model: {best_model}")

        # Visualization
        st.subheader("Visual Comparison")
        st.pyplot(comparator.plot_metric_comparison(comparison_df, metric_to_rank))

# Page: AutoML
elif page == "AutoML":
    st.markdown('<div class="sub-header">🤖 AutoML - Automated Model Selection</div>', unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
    else:
        df = st.session_state.data

        col1, col2 = st.columns(2)

        with col1:
            task_type = st.selectbox("Task Type", ["classification", "regression"])
            target_column = st.selectbox("Target Column", df.columns)

        with col2:
            n_trials = st.slider("Trials per Algorithm", 10, 100, 30)

        if st.button("🚀 Run AutoML"):
            with st.spinner("Running AutoML... This may take a few minutes."):
                try:
                    # Prepare data
                    X = df.drop(target_column, axis=1)
                    y = df[target_column]

                    # Preprocess
                    engineer = FeatureEngineer(scaling='standard', encoding='onehot')
                    X_processed = engineer.fit_transform(X, y)

                    # Run AutoML
                    automl = AutoMLOptimizer(
                        task=task_type,
                        n_trials_per_model=n_trials
                    )

                    best_model = automl.fit(X_processed, y)

                    st.success(f"✅ AutoML Complete!")

                    # Results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Best Algorithm", automl.best_algorithm)
                    with col2:
                        st.metric("Best Score", f"{automl.best_score:.4f}")

                    # Leaderboard
                    st.subheader("Algorithm Leaderboard")
                    st.dataframe(automl.get_leaderboard())

                    # Store best model
                    st.session_state.models['AutoML_Best'] = best_model
                    st.session_state.results['AutoML_Best'] = {'score': automl.best_score}

                except Exception as e:
                    st.error(f"Error: {e}")

# Page: About
elif page == "About":
    st.markdown('<div class="sub-header">ℹ️ About</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Unified AI Analytics Platform

    A comprehensive machine learning platform for automated model training,
    evaluation, and deployment.

    ### Features
    - 📁 Easy data upload and exploration
    - 🎯 Multiple ML algorithms (17+ models)
    - 📊 Comprehensive model evaluation
    - 🤖 AutoML for automatic model selection
    - 📈 Interactive visualizations
    - ⚡ Fast and efficient processing

    ### Supported Algorithms

    **Classification:**
    - Logistic Regression
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - SVM
    - KNN
    - Naive Bayes

    **Regression:**
    - Linear Regression
    - Ridge, Lasso, ElasticNet
    - Random Forest
    - XGBoost
    - LightGBM
    - CatBoost
    - SVM Regressor

    ### Tech Stack
    - **Backend**: Python, scikit-learn, XGBoost, LightGBM, CatBoost
    - **Frontend**: Streamlit
    - **Optimization**: Optuna
    - **Visualization**: Matplotlib, Seaborn

    ---

    Built with ❤️ using Claude Code
    """)

# Footer
st.markdown("---")
st.markdown("**Unified AI Analytics Platform** | Version 1.0.0")
