# 💻 ML Assignment 2 — Laptop Price Classification

> **Multi-class classification** of laptops into **Low**, **Medium**, and **High** price categories using six machine learning models.

**Name:** Rakesh R  
**BITS ID:** 2024dc04070  
**Course:** Machine Learning  
**Institution:** BITS Pilani  

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)

---

## Section A: Problem Statement

The goal of this project is to build and compare **six machine learning classification algorithms** to predict laptop price categories derived from continuous prices via quantile-based binning.

**Objective:**
- Perform **multi-class classification** into **Low**, **Medium**, and **High** price categories.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against ensemble methods (Random Forest, XGBoost).
- Ensure balanced classes using `pd.qcut` on the continuous price data.

---

## Section B: Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | [Kaggle Dataset Link](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |
| **Split** | 80 / 20 stratified |

### Feature Details
- **Categorical**: Company, TypeName, Cpu_brand, Gpu_brand, Os
- **Numerical**: Ram (GB), Weight (kg), TouchScreen (0/1), Ips (0/1), Ppi, HDD (GB), SSD (GB)

---

## Section C: Models Used & Performance Comparison

### 1. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|:--------------|:--------:|:---:|:---------:|:------:|:--:|:---:|
| **XGBoost (Ensemble)** | **0.8745** | **0.9421** | 0.8771 | 0.8745 | 0.8753 | 0.8122 |
| **Random Forest (Ensemble)** | 0.8471 | 0.9158 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| **Logistic Regression** | 0.8078 | 0.8812 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| **Decision Tree** | 0.8078 | 0.8545 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| **KNN** | 0.8039 | 0.8421 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| **Naive Bayes** | 0.5137 | 0.6542 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

### 2. Performance Observations
| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| **Logistic Regression** | Provides a strong baseline. It handles the linearly separable aspects of the data well. |
| **Decision Tree** | Efficient and easy to interpret. Comparable to Logistic Regression but more prone to variance. |
| **KNN** | Performs well by identifying similar laptop configurations based on nearest neighbors. |
| **Naive Bayes** | The weakest performer (51.37%) due to assumptions of feature independence. |
| **Random Forest (Ensemble)** | Significantly improves performance over a single tree by using bagging to reduce overfitting. |
| **XGBoost (Ensemble)** | The top performer (87.45%). Boosting sequentially minimizes errors for high accuracy. |

---

## 📁 Repository Structure
```
ML Assingment 2/
├── .devcontainer/                  # Development configuration
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── train_models.py                 # Standalone training script
├── app.py                          # Streamlit web application
├── laptop_data_cleaned.csv         # Cleaned dataset
├── model_comparison_metrics.csv    # Evaluation results
└── all_models.pkl                  # All 6 trained model pipelines
```

---

## 🚀 Getting Started
```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🖥️ Streamlit App Features
- **Dataset upload option (CSV)**: Predict for custom data files.
- **Model selection**: Choose between all 6 implemented models.
- **Evaluation metrics**: Real-time display of performance scores.
- **Performance Heatmap**: Visual breakdown of model accuracy across metrics.

---

## 🛠️ Technologies Used
- **ML Framework**: Scikit-learn, XGBoost
- **Data**: Pandas, NumPy
- **Visuals**: Plotly, Seaborn
- **Web App**: Streamlit

---

**Submission Date:** February 15, 2026
