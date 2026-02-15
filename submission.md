# Assignment Submission: Laptop Price Predictor

**Student Name:** Rakesh R  
**BITS ID:** 2024dc04070  
**Course:** Machine Learning (BITS Pilani)  
**Submission Date:** February 15, 2026

---

## 1. GitHub Repository Link
[https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-](https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-)  

**Repository Contents:**
- Complete source code (`app.py`, `train_models.py`, etc.)
- Dependencies (`requirements.txt`)
- Comprehensive Project Documentation (`README.md`)

---

## 2. Live Streamlit App Link
[https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)  
*(Deployed using Streamlit Community Cloud — Interactive Frontend)*

---

## 3. Screenshot of Assignment Execution (BITS Virtual Lab)

![BITS Virtual Lab Execution Screenshot](placeholder_for_screenshot.png)

---

## 4. GitHub README Content

# 💻 Laptop Price Classification & Prediction

> **Multi-class classification** of laptops into **Low**, **Medium**, and **High** price categories using six machine learning models.

---

### Section A: Problem Statement & Objectives
The goal of this project is to build and compare six machine learning classification algorithms to predict laptop price categories. The continuous price data was transformed into discrete categories (**Low**, **Medium**, **High**) using quantile-based binning to ensure balanced classes.

**Key Objectives:**
- Perform **multi-class classification** on structured laptop specifications.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against advanced ensemble methods (Random Forest, XGBoost).
- Deploy an interactive **Streamlit** dashboard for real-time predictions and model evaluation.

---

### Section B: Dataset Description
| Property | Detail |
|----------|--------|
| **Source** | [Kaggle Dataset](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |
| **Split** | 80 / 20 stratified |

**Feature Breakdown:**
- **Categorical**: Company, TypeName, Cpu_brand, Gpu_brand, Os
- **Numerical**: Ram (GB), Weight (kg), TouchScreen (0/1), Ips (0/1), Ppi, HDD (GB), SSD (GB)

---

### Section C: Models Used & Comparative Analysis

#### 1. Performance Metrics
| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|:--------------|:--------:|:---------:|:---------:|:------:|:--------:|:---:|
| **XGBoost (Ensemble)** | **0.8745** | **0.9421** | 0.8771 | 0.8745 | 0.8753 | 0.8122 |
| **Random Forest (Ensemble)** | 0.8471 | 0.9158 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| **Logistic Regression** | 0.8078 | 0.8812 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| **Decision Tree** | 0.8078 | 0.8545 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| **KNN** | 0.8039 | 0.8421 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| **Naive Bayes** | 0.5137 | 0.6542 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

#### 2. Observation Table
| ML Model Name | Observation about model performance |
|:--------------|:------------------------------------|
| **Logistic Regression** | Provides a strong baseline. Efficiently identifies linear relationships in hardware specs. |
| **Decision Tree** | Interpretable but less stable; accuracy is comparable to baseline methods. |
| **kNN** | Effective at local pattern matching based on similar laptop configurations. |
| **Naive Bayes** | Weakest performer; the independence assumption is violated by correlated features like RAM and CPU brand. |
| **Random Forest (Ensemble)** | Significant boost in stability and accuracy using the bagging technique. |
| **XGBoost (Ensemble)** | **Top performer** (87.5% Accuracy); gradient boosting effectively minimizes residual errors. |

---

### 🖥️ Streamlit App Features
- **Project upload option (CSV)**: Batch prediction support for test data files.
- **Sample Test Data**: Ready-to-use sample CSV download for easy evaluation.
- **Model selection**: Choose between 6 diverse ML models via dropdown.
- **Evaluation metrics**: Live display of Accuracy, AUC, and F1.
- **Visual Evidence**: Integrated **Confusion Matrix** (Heatmap) and **Classification Report**.

---

### 🛠️ Tools Used
- **Languages**: Python
- **ML Frameworks**: Scikit-learn, XGBoost
- **Dashboards**: Streamlit
- **Data/Visuals**: Pandas, Plotly, Seaborn
