from pathlib import Path
import pandas as pd
import numpy as np

# loading the dataset
file_path = Path(r"C:\Users\win 10\Desktop\ml project\heart.csv")
df = pd.read_csv(file_path)
print("Original dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

#checking data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

#checking for missing values
print("\n========== MISSING VALUES ==========")

print(df.isnull().sum())

print("\nTotal missing values:",
      df.isnull().sum().sum())

#checking for duplicates
print("\n========== DUPLICATES ==========") 
duplicate_count = df.duplicated().sum()
print("Duplicate rows:", duplicate_count)
print("Unique rows:", df.drop_duplicates().shape[0])

#removing exact duplicates
df = df.drop_duplicates().reset_index(drop=True)
print("\nShape after removing duplicates:", df.shape)

#checking target values
print("\n========== TARGET DISTRIBUTION ==========")
print(df["target"].value_counts())
print("\nTarget percentage:")
print(
    df["target"].value_counts(normalize=True) * 100
)

#checking invalid values in numerical columns
print("\n========== INVALID VALUES ==========")

# Check whether numerical values are negative
numeric_columns = df.select_dtypes(
    include=np.number
).columns
for column in numeric_columns:

    negative_count = (df[column] < 0).sum()

    print(
        column,
        "negative values:",
        negative_count
    )

# checking for outliers using IQR method
print("\n========== OUTLIERS ==========")

numeric_cols = df.select_dtypes(include=["number"]).columns

for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

    print(f"{col}: {len(outliers)} outliers")
    print(f"  Lower bound: {lower_bound}")
    print(f"  Upper bound: {upper_bound}")

#checking target class
print("\nTarget unique values:")
print(df["target"].unique())

#statistical summary of the dataset
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())    

# saving the cleaned dataset to a new CSV file
cleaned_file_path = Path(r"C:\Users\win 10\Desktop\ml project\cleaned_heart.csv")
df.to_csv(cleaned_file_path, index=False)







