# Assignment Submission: Laptop Price Predictor

**Name:** Rakesh R  
**BITS ID:** 2024dc04070  
**Course:** Machine Learning  
**Institution:** BITS Pilani  

---

## 1. Submission Links

**GitHub Repository:** [https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-](https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-)  
**Live Streamlit App:** [https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)

---

## 2. Screenshot of Execution (BITS Virtual Lab)

![BITS Virtual Lab Screenshot](placeholder_for_screenshot.png)

---

## 3. Project Documentation (README)

# 💻 Laptop Price Classification Project

---

### Section A: Problem Statement

The goal of this project is to build and compare **six machine learning classification algorithms** to predict laptop price categories derived from continuous prices via quantile-based binning.

**Objective:**
- Perform **multi-class classification** into **Low**, **Medium**, and **High** price categories.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against ensemble methods (Random Forest, XGBoost).

---

### Section B: Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | [Kaggle Dataset Link](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |

**Feature Details:**
- **Categorical**: Company, TypeName, Cpu_brand, Gpu_brand, Os
- **Numerical**: Ram (GB), Weight (kg), TouchScreen (0/1), Ips (0/1), Ppi, HDD (GB), SSD (GB)

---

### Section C: Models & Performance Comparison

#### 1. Performance Metrics Table
| ML Model Name | Accuracy | Precision | Recall | F1 Score | MCC |
|:--------------|:--------:|:---------:|:------:|:--------:|:---:|
| **XGBoost (Ensemble)** | **0.8745** | **0.8771** | **0.8745** | **0.8753** | **0.8122** |
| **Random Forest (Ensemble)** | 0.8471 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| **Logistic Regression** | 0.8078 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| **Decision Tree** | 0.8078 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| **KNN** | 0.8039 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| **Naive Bayes** | 0.5137 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

#### 2. Performance Observations
| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| **Logistic Regression** | Strong baseline; handles linearly separable data well. |
| **Decision Tree** | Interpretable; comparable accuracy to linear models but higher variance. |
| **KNN** | Effective using nearest neighbors for similar laptop specs. |
| **Naive Bayes** | Weakest performer due to feature independence assumptions. |
| **Random Forest (Ensemble)** | Robust performance through bagging and reduced overfitting. |
| **XGBoost (Ensemble)** | Top performer; sequential boosting effectively minimizes errors. |

---

### 🚀 Additional Project Details

#### Streamlit App Features
- **Visual Comparison**: Interactive Plotly charts for model leaderboard.
- **Micro-Animations**: Gradient banners and metric cards for a premium UI.
- **Batch Processing**: CSV upload capability for bulk price prediction.

#### Technologies Used
- Scikit-learn, XGBoost, Pandas, NumPy, Plotly, Streamlit.
