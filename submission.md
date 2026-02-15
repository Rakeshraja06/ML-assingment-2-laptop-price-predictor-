# Assignment Submission: Laptop Price Predictor

## 1. GitHub Repository Link
**Link:** [https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-](https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-)

The repository contains:
- **Complete Source Code**
- **requirements.txt**
- **Restructured README.md**

---

## 2. Live Streamlit App Link
**Link:** [https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)

---

## 3. Screenshot of Execution on BITS Virtual Lab

![BITS Virtual Lab Screenshot](placeholder_for_screenshot.png)

---

## 4. Project Documentation (README.md)

# 💻 ML Assignment 2 — Laptop Price Classification

---

## Section A: Problem Statement

The goal of this project is to build and compare **six machine learning classification algorithms** to predict laptop price categories. 

### Objective
- Perform **multi-class classification** into **Low**, **Medium**, and **High** price categories.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against ensemble methods (Random Forest, XGBoost).
- Derive price categories from continuous price data using quantile-based binning (`pd.qcut`) to ensure balanced classes.

---

## Section B: Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | [Laptop Price Prediction Cleaned Dataset (Kaggle)](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |
| **Split** | 80 / 20 stratified |

### Feature Details
- **Company**: Manufacturer (Apple, HP, Dell, ...)
- **TypeName**: Laptop type (Ultrabook, Gaming, ...)
- **Ram**: RAM in GB
- **Weight**: Weight in kg
- **TouchScreen**: Binary (0/1)
- **Ips**: IPS display (0/1)
- **Ppi**: Pixels per inch
- **Cpu_brand**: CPU brand
- **HDD**: HDD capacity (GB)
- **SSD**: SSD capacity (GB)
- **Gpu_brand**: GPU brand (Intel, Nvidia, AMD)
- **Os**: Operating system

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
| **Logistic Regression** | Provides a strong baseline. It handles the linearly separable aspects of the data well but is slightly outperformed by tree-based models. |
| **Decision Tree** | Efficient and easy to interpret. It achieves comparable accuracy to Logistic Regression but is naturally more prone to variance compared to ensembles. |
| **KNN** | Performs well by identifying similar laptop configurations. Its performance is on par with the basic tree and linear models for this dataset. |
| **Naive Bayes** | The weakest performer (51.37%). This is likely due to its strong assumption of feature independence, which is violated by the high correlation between specs like RAM, CPU, and Price. |
| **Random Forest (Ensemble)** | Significantly improves performance over a single Decision Tree by reducing overfitting through bagging, resulting in a robust 84.71% accuracy. |
| **XGBoost (Ensemble)** | The top performer (87.45%). Its sequential boosting approach effectively minimizes errors that other models miss, making it the most reliable for this classification task. |

---

## 👤 Author 
**Rakesh R** (2024dc04070)  
BITS Pilani — Machine Learning, Assignment 2
Submission Date: February 15, 2026
