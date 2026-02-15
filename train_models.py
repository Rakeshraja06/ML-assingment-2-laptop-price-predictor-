
import numpy as np
import pandas as pd
import os
import joblib
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
    matthews_corrcoef,
    roc_auc_score
)

# 1. Load Dataset
file_name = "laptop_data_cleaned.csv"
# df already exists

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
    
    try:
        if name == "XGBoost":
             # XGBoost requires numerical labels
            y_train_encoded = y_train.cat.codes
            pipeline.fit(X_train, y_train_encoded)
            y_pred_numeric = pipeline.predict(X_test)
            y_pred = original_categories[y_pred_numeric]
        else:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
    except Exception as e:
        print(f"Skipping {name} due to error: {e}")
        continue
        
    # Calculate probabilities for AUC
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)
    else:
        # For models without predict_proba (like some Naive Bayes cases, though GaussianNB has it)
        # we can use decison_function if available, or just skip if necessary
        y_prob = None

    # AUC calculation (multi-class OVR)
    auc_score = 0
    if y_prob is not None:
        try:
            # XGBoost encoded labels, so y_test needs to match types for roc_auc_score
            y_test_numeric = y_test.cat.codes
            auc_score = roc_auc_score(y_test_numeric, y_prob, multi_class="ovr", average="weighted")
        except Exception as e:
            print(f"Warning: Could not calculate AUC for {name}: {e}")
            auc_score = 0

    # Metrics
    results.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC Score": auc_score,
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
