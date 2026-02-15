
import numpy as np
import pandas as pd
import os
import joblib
import gdown
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# 1. Load Dataset
file_name = "laptop_data_cleaned.csv"
if not os.path.exists(file_name):
    # Direct download link
    direct_download_url = 'https://drive.google.com/uc?id=1W5Hgc8HTOm1ShOS0Bt_0Dwtq1scS5HmV'
    print(f"Downloading '{file_name}' from Google Drive...")
    gdown.download(direct_download_url, output=file_name, quiet=False)
else:
    print(f"'{file_name}' already exists.")

df = pd.read_csv(file_name)

# 2. Convert Regression -> Classification
df["Price_Category"] = pd.qcut(
    df["Price"],
    q=3,
    labels=["Low", "Medium", "High"]
)

# 3. Split Features & Target
X = df.drop(["Price", "Price_Category"], axis=1)
y = df["Price_Category"]

# 4. Preprocessing
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ]
)

# 5. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Define Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42
    )
}

# 7. Train, Evaluate & Save
results = []
all_pipelines = {}
original_categories = y.cat.categories


print("\nTraining models...")
for name, model in models.items():
    print(f"Training {name}...")
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    if name == "XGBoost":
         # XGBoost requires numerical labels
        y_train_encoded = y_train.cat.codes
        pipeline.fit(X_train, y_train_encoded)
        y_pred_numeric = pipeline.predict(X_test)
        y_pred = original_categories[y_pred_numeric]
    else:
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
    # Metrics
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted"),
        "Recall": recall_score(y_test, y_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred, average="weighted"),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })
    
    all_pipelines[name] = pipeline
    print(f"Trained {name} ✓")

# Save all models in a single pkl file
model_path = "all_models.pkl"
joblib.dump(all_pipelines, model_path)
print(f"\nAll models saved to {model_path}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison_metrics.csv", index=False)
print("\nModel comparison saved to model_comparison_metrics.csv")
print(results_df)
