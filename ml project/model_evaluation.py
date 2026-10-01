#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
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
    classification_report
)

#loading the dataset
df = pd.read_csv(r"C:\Users\win 10\Desktop\ml project\cleaned_heart.csv")
print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

#separating features and target
X = df.drop("target", axis=1)
y = df["target"]
print("\nFeatures:")
print(X.columns.tolist())
print("\nTarget distribution:")
print(y.value_counts())

#splitting the dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

#defining models for evaluation
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

#training and evaluating each model
results = []
trained_models = {}
for name, model in models.items():
    print("\n" + "=" * 50)
    print("Training:", name)
    print("=" * 50)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Get probability predictions
    y_probability = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
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

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc
    })

    # Store trained model
    trained_models[name] = model

    # Display classification report
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

#creating a table to display the evaluation metrics
results_df = pd.DataFrame(results)
print("\n")
print("=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)
print(results_df.round(4))

# Save results table as CSV
results_df.to_csv(
    "model_comparison_results.csv",
    index=False
)
print("\nResults table saved as:")
print("model_comparison_results.csv")

#creating a bar plot to visualize the evaluation metrics
metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

ax = results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(13, 7)
)

ax.set_title("Comparison of Machine Learning Models")
ax.set_xlabel("Machine Learning Models")
ax.set_ylabel("Performance Score")
ax.set_ylim(0, 1)

plt.xticks(rotation=30)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

plt.savefig(
    "model_comparison_bar_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

#creating a confusion matrix for each model
for name, model in trained_models.items():

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title("Confusion Matrix - " + name)
    plt.xlabel("Predicted Value")
    plt.ylabel("Actual Value")
    plt.tight_layout()

    # Create a safe filename
    filename = name.lower().replace(" ", "_")

    # Save confusion matrix
    plt.savefig(
        "confusion_matrix_" + filename + ".png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()

#recommendations based on evaluation metrics
# Sort models by Recall first and F1-Score second
# Recall is prioritized for identifying positive cases.

ranked_results = results_df.sort_values(
    by=["Recall", "F1-Score"],
    ascending=False
)

recommended_model = ranked_results.iloc[0]
print("\n")
print("=" * 70)
print("TECHNICAL MODEL RECOMMENDATION")
print("=" * 70)

print(
    "Model selected according to Recall and F1-Score:",
    recommended_model["Model"]
)
print("Recall:", round(recommended_model["Recall"], 4))
print("F1-Score:", round(recommended_model["F1-Score"], 4))
print(
    "\nNote: This recommendation is based only on this dataset "
    "and the selected evaluation criteria."
)
print(
    "It is not a clinically validated medical recommendation."
)