#importing useful libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# loading cleaned dataset
df = pd.read_csv("C:\\Users\\win 10\\Desktop\\ml project\\cleaned_heart.csv")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

#seperate features and target variable
X = df.drop("target", axis=1)
y = df["target"]
print("\n========== FEATURES AND TARGET ==========")
print("Features shape:", X.shape)
print("Target shape:", y.shape)

# splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== DATA SPLIT ==========")
print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# creating a preprocessing models pipeline
models = {
    # Logistic Regression
    # Scaling is applied inside the pipeline
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000
        ))
    ]),

    # Support Vector Machine
    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            probability=True
        ))
    ]),

    # K-Nearest Neighbors
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),

    # Decision Tree
    # Scaling is not required
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    # Random Forest
    # Scaling is not required
    "Random Forest": RandomForestClassifier(
        random_state=42
    )
}

# displaying the models
print("\n========== MODELS CREATED ==========")
for name in models:
    print(name)

# training the models
print("\n========== TRAINING MODELS ==========")
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "training completed")

# verifying prediction
print("\n========== TEST PREDICTIONS ==========")
for name, model in models.items():
    predictions = model.predict(X_test)
    print(
        name,
        "predictions generated:",
        len(predictions)
    )

# summary
print("\n========== PREPROCESSING COMPLETED ==========")
print("All models trained successfully.")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Number of models:", len(models))
