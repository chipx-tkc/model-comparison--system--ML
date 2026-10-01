import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    learning_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    auc
)
# PAGE CONFIGURATION
st.set_page_config(
    page_title="MECS Dashboard",
    page_icon="🤖",
    layout="wide"
)
# CUSTOM STYLING
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
}
.subtitle {
    font-size: 20px;
    text-align: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
# TITLE
st.markdown(
    '<div class="main-title">🤖 MECS Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Model Evaluation and Comparison System</div>',
    unsafe_allow_html=True
)
st.divider()
# SIDEBAR
st.sidebar.title("⚙️ Dashboard Controls")
st.sidebar.write(
    "Use the controls below to explore the machine learning models."
)
# Model selection
model_names = [
    "All Models",
    "Logistic Regression",
    "SVM",
    "KNN",
    "Decision Tree",
    "Random Forest"
]
selected_model = st.sidebar.selectbox(
    "🔍 Select Model",
    model_names
)
# Dataset visibility
show_dataset = st.sidebar.checkbox(
    "📄 Show Dataset",
    value=True
)
# Metric selection
selected_metrics = st.sidebar.multiselect(
    "📊 Select Metrics",
    [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ],
    default=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC"
    ]
)
# LOAD DATASET
df = pd.read_csv(
    r"C:\Users\win 10\Desktop\ml project\cleaned_heart.csv"
)
# Check target column
if "target" not in df.columns:
    st.error(
        "The dataset must contain a column named 'target'."
    )
    st.stop()
st.sidebar.success("Dataset loaded successfully!")
# PREPARE DATA
X = df.drop("target", axis=1)
y = df["target"]
# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# DEFINE MODELS
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(max_iter=1000)
        )
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            SVC(probability=True)
        )
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            KNeighborsClassifier()
        )
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}
# TRAIN ALL MODELS
results = []
trained_models = {}
predictions = {}
probabilities = {}
for name, model in models.items():
    model.fit(
        X_train,
        y_train
    )
    # Store trained model
    trained_models[name] = model
    # Predictions
    y_pred = model.predict(X_test)
    # Probabilities
    y_probability = model.predict_proba(X_test)[:, 1]
    # Store predictions
    predictions[name] = y_pred
    probabilities[name] = y_probability
    # Metrics
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )
    # Save results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })
# Results DataFrame
results_df = pd.DataFrame(results)
# TABS
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Overview",
    "📊 Dataset",
    "🤖 Model Performance",
    "📈 Visualizations",
    "🔬 Advanced Analysis",
    "💡 Insights"
])
# TAB 1 — OVERVIEW
with tab1:
    st.header("🏠 MECS Dashboard Overview")
    st.write(
        """
        The Model Evaluation and Comparison System (MECS)
        evaluates multiple machine learning classification models,
        compares their performance using different evaluation
        metrics, and provides a technical model recommendation.
        """
    )
    st.subheader("📌 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Total Records",
            df.shape[0]
        )
    with col2:
        st.metric(
            "Features",
            df.shape[1] - 1
        )
    with col3:
        st.metric(
            "Missing Values",
            df.isnull().sum().sum()
        )
    with col4:
        st.metric(
            "Models Evaluated",
            len(models)
        )
    st.divider()
    st.subheader("🤖 Models Evaluated")
    model_table = pd.DataFrame({
        "Model": list(models.keys()),
        "Type": [
            "Linear",
            "Kernel-based",
            "Instance-based",
            "Tree-based",
            "Ensemble"
        ]
    }
    )
    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )
# TAB 2 — DATASET
with tab2:
    st.header("📊 Dataset Exploration")
    if show_dataset:
        st.subheader("📄 Dataset Preview")
        st.dataframe(
            df,
            use_container_width=True
        )
    st.subheader("📐 Dataset Shape")
    st.write(
        f"Rows: **{df.shape[0]}**"
    )
    st.write(
        f"Columns: **{df.shape[1]}**"
    )
    st.subheader("📋 Column Names")
    st.write(
        df.columns.tolist()
    )
    st.subheader("📈 Statistical Summary")
    st.dataframe(
        df.describe(),
        use_container_width=True
    )
    st.subheader("🎯 Target Distribution")
    target_counts = df["target"].value_counts()
    st.bar_chart(
        target_counts
    )
    st.subheader("🔎 Missing Values")
    missing_values = pd.DataFrame({
        "Column": df.columns,
        "Missing Values":
            df.isnull().sum().values
    })
    st.dataframe(
        missing_values,
        use_container_width=True,
        hide_index=True
    )
# TAB 3 — MODEL PERFORMANCE
with tab3:
    st.header("🤖 Model Performance")
    st.subheader("📊 Model Comparison Results")
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-Score": "{:.3f}",
            "ROC-AUC": "{:.3f}"
        }),
        use_container_width=True
    )

    st.subheader("🔍 Selected Model Details")
    if selected_model == "All Models":
        st.info(
            "Showing results for all models."
        )
        selected_result = results_df
    else:
        selected_result = results_df[
            results_df["Model"] == selected_model
        ]
        st.write(
            f"Performance of **{selected_model}**:"
        )
        st.dataframe(
            selected_result.style.format({
                "Accuracy": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-Score": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True
        )
        metric1, metric2, metric3, metric4 = st.columns(4)
        with metric1:
            st.metric(
                "Accuracy",
                f"{selected_result['Accuracy'].iloc[0]:.3f}"
            )
        with metric2:
            st.metric(
                "Precision",
                f"{selected_result['Precision'].iloc[0]:.3f}"
            )
        with metric3:
            st.metric(
                "Recall",
                f"{selected_result['Recall'].iloc[0]:.3f}"
            )
        with metric4:
            st.metric(
                "F1-Score",
                f"{selected_result['F1-Score'].iloc[0]:.3f}"
            )
    # Interactive model comparison
    st.subheader("⚖️ Interactive Model Comparison")
    comparison_models = st.multiselect(
        "Select models to compare",
        list(models.keys()),
        default=list(models.keys())
    )
    if comparison_models:
        comparison_data = results_df[
            results_df["Model"].isin(comparison_models)
        ]
        st.bar_chart(
            comparison_data.set_index("Model")[
                [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1-Score",
                    "ROC-AUC"
                ]
            ]
        )
# TAB 4 — VISUALIZATIONS
with tab4:
    st.header("📈 Model Visualizations")
    st.subheader("📊 Model Performance Comparison")
    if len(selected_metrics) == 0:
        st.warning(
            "Select at least one metric from the sidebar."
        )
    else:
        chart_data = results_df.set_index("Model")[
            selected_metrics
        ]
        st.bar_chart(
            chart_data
        )

    st.subheader("🔲 Confusion Matrix")
    if selected_model == "All Models":
        models_to_display = list(
            trained_models.keys()
        )
    else:
        models_to_display = [
            selected_model
        ]
    for name in models_to_display:
        cm = confusion_matrix(
            y_test,
            predictions[name]
        )
        fig, ax = plt.subplots(
            figsize=(5, 4)
        )
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax
        )
        ax.set_title(
            f"Confusion Matrix - {name}"
        )
        ax.set_xlabel(
            "Predicted Value"
        )
        ax.set_ylabel(
            "Actual Value"
        )
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("📈 ROC Curves")
    fig, ax = plt.subplots(
        figsize=(8, 6)
    )
    if selected_model == "All Models":
        roc_models = trained_models.keys()
    else:
        roc_models = [selected_model]
    for name in roc_models:
        fpr, tpr, _ = roc_curve(
            y_test,
            probabilities[name]
        )
        roc_auc_value = auc(
            fpr,
            tpr
        )
        ax.plot(
            fpr,
            tpr,
            label=f"{name} (AUC = {roc_auc_value:.3f})"
        )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )
    ax.set_xlabel(
        "False Positive Rate"
    )
    ax.set_ylabel(
        "True Positive Rate"
    )
    ax.set_title(
        "ROC Curve Comparison"
    )
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("🌳 Feature Importance")
    if selected_model in [
        "Decision Tree",
        "Random Forest"
    ]:
        model_object = trained_models[
            selected_model
        ]
        importances = (
            model_object.feature_importances_
        )
        importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(
            by="Importance",
            ascending=False
        )
        fig, ax = plt.subplots(
            figsize=(8, 6)
        )
        sns.barplot(
            data=importance_df,
            x="Importance",
            y="Feature",
            ax=ax
        )
        ax.set_title(
            f"Feature Importance - {selected_model}"
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info(
            "Select Decision Tree or Random Forest "
            "from the sidebar to view feature importance."
        )

# TAB 5 — ADVANCED ANALYSIS
with tab5:
    st.header("🔬 Advanced Model Analysis")
    # Cross Validation
    st.subheader("🔄 5-Fold Cross-Validation")
    cv_results = []
    for name, model in models.items():
        scores = cross_val_score(
            model,
            X,
            y,
            cv=5,
            scoring="f1"
        )
        cv_results.append({
            "Model": name,
            "Mean F1": scores.mean(),
            "Standard Deviation": scores.std()
        })
    cv_df = pd.DataFrame(
        cv_results
    )
    st.dataframe(
        cv_df.style.format({
            "Mean F1": "{:.3f}",
            "Standard Deviation": "{:.3f}"
        }),
        use_container_width=True
    )
    # Learning Curve
    st.subheader("📚 Learning Curve")
    learning_model_name = st.selectbox(
        "Select Model for Learning Curve",
        list(models.keys())
    )
    learning_model = models[
        learning_model_name
    ]
    train_sizes, train_scores, validation_scores = learning_curve(
        learning_model,
        X,
        y,
        cv=5,
        scoring="f1",
        train_sizes=np.linspace(
            0.1,
            1.0,
            5
        )
    )
    train_mean = train_scores.mean(
        axis=1
    )
    validation_mean = validation_scores.mean(
        axis=1
    )
    fig, ax = plt.subplots(
        figsize=(8, 5)
    )
    ax.plot(
        train_sizes,
        train_mean,
        marker="o",
        label="Training F1"
    )
    ax.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Validation F1"

    )
    ax.set_xlabel(
        "Training Examples"
    )
    ax.set_ylabel(
        "F1-Score"
    )
    ax.set_title(
        f"Learning Curve - {learning_model_name}"
    )
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.info(
        "The learning curve helps examine how model "
        "performance changes as more training data is used."
    )
# TAB 6 — INSIGHTS
with tab6:
    st.header("💡 Model Insights")
    # Technical Recommendation
    st.subheader(
        "🏆 Technical Model Recommendation"
    )
    ranked_results = results_df.sort_values(
        by=[
            "Recall",
            "F1-Score"
        ],
        ascending=False
    )
    recommended_model = ranked_results.iloc[0]
    st.success(
        f"Selected Model: "
        f"{recommended_model['Model']}"
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Recall",
            f"{recommended_model['Recall']:.3f}"
        )
    with col2:
        st.metric(
            "F1-Score",
            f"{recommended_model['F1-Score']:.3f}"
        )
    with col3:
        st.metric(
            "Accuracy",
            f"{recommended_model['Accuracy']:.3f}"
        )
    with col4:
        st.metric(
            "ROC-AUC",
            f"{recommended_model['ROC-AUC']:.3f}"
        )
    st.info(
        "The technical selection uses Recall as the primary "
        "criterion and F1-Score as the secondary criterion. "
        "This is an evaluation of the models on this dataset "
        "and is not a clinically validated medical recommendation."
    )
    # Model Explanations
    st.subheader(
        "🧠 Model Explanations"
    )
    explanations = {
        "Logistic Regression":
            "A linear classification algorithm that estimates the probability of a class.",
        "SVM":
            "A classification algorithm that identifies a decision boundary between classes.",
        "KNN":
            "A distance-based algorithm that classifies observations using nearby observations.",
        "Decision Tree":
            "A tree-based algorithm that uses a sequence of decision rules.",
        "Random Forest":
            "An ensemble method that combines multiple decision trees."
    }
    for name, explanation in explanations.items():
        with st.expander(name):
            st.write(
                explanation
            )
   
    # Download Results
    st.subheader(
        "📥 Download Results"
    )
    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")
    st.download_button(
        label="📥 Download Evaluation Results",
        data=csv_data,
        file_name="model_evaluation_results.csv",
        mime="text/csv"
    )
# FOOTER
st.divider()
st.caption(
    "MECS Dashboard | Machine Learning Model Evaluation and Comparison System"
)