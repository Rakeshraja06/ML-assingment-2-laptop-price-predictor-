# 💻 ML Assignment 2 — Laptop Price Classification

> **Multi-class classification** of laptops into **Low**, **Medium**, and **High** price categories using six machine learning models, including **ensemble methods** (Random Forest & XGBoost).

**Name:** Rakesh R  
**BITS ID:** 2024dc04070  
**Course:** Machine Learning  
**Institution:** BITS Pilani  

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)

---

## Section A: Problem Statement

The goal of this project is to build and compare **six machine learning classification algorithms** to predict laptop price categories derived from continuous prices via quantile-based binning.

### Objective
- Perform **multi-class classification** into **Low**, **Medium**, and **High** price categories.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against ensemble methods (Random Forest, XGBoost).
- Ensure balanced classes using `pd.qcut` on the continuous price data.

---

## Section B: Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | [Laptop Price Prediction Cleaned Dataset (Kaggle)](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |
| **Split** | 80 / 20 stratified |

<details>
<summary><strong>📑 Feature Details (click to expand)</strong></summary>

| # | Feature | Type | Description |
|---|---------|------|-------------|
| 1 | `Company` | Categorical | Manufacturer (Apple, HP, Dell, …) |
| 2 | `TypeName` | Categorical | Laptop type (Ultrabook, Gaming, …) |
| 3 | `Ram` | Numerical | RAM in GB |
| 4 | `Weight` | Numerical | Weight in kg |
| 5 | `TouchScreen` | Binary | 0 / 1 |
| 6 | `Ips` | Binary | IPS display (0 / 1) |
| 7 | `Ppi` | Numerical | Pixels per inch |
| 8 | `Cpu_brand` | Categorical | CPU brand |
| 9 | `HDD` | Numerical | HDD capacity (GB) |
| 10 | `SSD` | Numerical | SSD capacity (GB) |
| 11 | `Gpu_brand` | Categorical | GPU brand (Intel, Nvidia, AMD) |
| 12 | `Os` | Categorical | Operating system |

</details>

---

## Section C: Models Used & Performance Comparison

### 1. Model Comparison Table
| ML Model Name | Accuracy | Precision | Recall | F1 Score | MCC |
|:--------------|:--------:|:---------:|:------:|:--------:|:---:|
| **XGBoost (Ensemble)** | **0.8745** | **0.8771** | **0.8745** | **0.8753** | **0.8122** |
| **Random Forest (Ensemble)** | 0.8471 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| **Logistic Regression** | 0.8078 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| **Decision Tree** | 0.8078 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| **KNN** | 0.8039 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| **Naive Bayes** | 0.5137 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

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
├── .devcontainer/                  # Development container configuration
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

### Installation
```bash
# 1. Clone & Install
pip install -r requirements.txt

# 2. Run App
streamlit run app.py
```

---

## 🖥️ Streamlit App Features

| Feature | Description |
|---------|-------------|
| 🎨 **Rich UI** | Gradient banners and dark-themed metric cards |
| 📊 **Interactive Charts** | Plotly charts and heatmaps for model comparison |
| 🔮 **Manual Prediction** | Get instant price category prediction from specs |
| 📁 **Batch Prediction** | Upload CSV for bulk predictions |

---

## 🛠️ Technologies Used

- **Language**: Python 3.8+
- **ML Framework**: scikit-learn, XGBoost
- **Data**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Web App**: Streamlit
- **Deployment**: Streamlit Cloud

---

**Submission Date:** February 15, 2026
