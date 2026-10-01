#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the cleaned dataset 
df = pd.read_csv(r"C:\Users\win 10\Desktop\ml project\cleaned_heart.csv")
print("Dataset shape:", df.shape)
print("\nDataset information:")
df.info()

#target distribution
print("\n========== TARGET DISTRIBUTION ==========")
print(df["target"].value_counts())
print("\nTarget percentages:")
print(
    df["target"].value_counts(normalize=True) * 100
)

plt.figure(figsize=(7, 5))
sns.countplot(
    data=df,
    x="target"
)

plt.title("Heart Disease Target Distribution")
plt.xlabel("Target (0 = No, 1 = Yes)")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.show()

#numerical features distribution
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()

df[numeric_columns].hist(
    figsize=(15, 12),
    bins=20
)
plt.suptitle(
    "Distribution of Numerical Features"
)
plt.tight_layout()
plt.show()

#correlation heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Heart Disease Feature Correlation")
plt.tight_layout()
plt.show()

#boxplots for numerical features
plt.figure(figsize=(15, 8))
df[numeric_columns].boxplot(
    rot=45
)

plt.title("Numerical Feature Boxplots")
plt.tight_layout()
plt.show()

#feature target relationship analysis
# Age vs target
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="target",
    y="age"
)

plt.title("Age Distribution by Target")
plt.xlabel("Target")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# Maximum heart rate vs target
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="target",
    y="thalach"
)

plt.title("Maximum Heart Rate by Target")
plt.xlabel("Target")
plt.ylabel("Maximum Heart Rate")
plt.tight_layout()
plt.show()

# Cholesterol vs target
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x="target",
    y="chol"
)

plt.title("Cholesterol Distribution by Target")
plt.xlabel("Target")
plt.ylabel("Cholesterol")
plt.tight_layout()
plt.show()

#gruop statistics for categorical features
print("\n========== GROUP STATISTICS ==========")
print(
    df.groupby("target")[
        ["age", "trestbps", "chol", "thalach", "oldpeak"]
    ].mean()
)

# final dataset summary
print("\n========== FINAL SUMMARY ==========")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:",
      df.isnull().sum().sum())
print("Duplicate rows:",
      df.duplicated().sum())