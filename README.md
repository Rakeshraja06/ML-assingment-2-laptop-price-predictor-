# 💻 Laptop Price Classification & Prediction

> **Multi-class classification** of laptops into **Low**, **Medium**, and **High** price categories using six machine learning models.

[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)

---

## 🎯 Project Overview
The goal of this project is to build and compare six machine learning classification algorithms to predict laptop price categories. The continuous price data was transformed into discrete categories (**Low**, **Medium**, **High**) using quantile-based binning to ensure balanced classes.

**Key Objectives:**
- Perform **multi-class classification** on structured laptop specifications.
- Compare traditional models (Logistic Regression, Decision Tree, KNN, Naive Bayes) against advanced ensemble methods (Random Forest, XGBoost).
- Deploy an interactive **Streamlit** dashboard for real-time predictions and model evaluation.

---

## 📊 Dataset Description
| Property | Detail |
|----------|--------|
| **Source** | [Kaggle Dataset](https://www.kaggle.com/datasets/gyanprakashkushwaha/laptop-price-prediction-cleaned-dataset) |
| **Instances** | 1,273 |
| **Features** | 12 (5 categorical, 7 numerical) |
| **Target** | `Price_Category` — Low · Medium · High |
| **Split** | 80 / 20 stratified |

### Feature Breakdown
- **Categorical**: Company, TypeName, Cpu_brand, Gpu_brand, Os
- **Numerical**: Ram (GB), Weight (kg), TouchScreen (0/1), Ips (0/1), Ppi, HDD (GB), SSD (GB)

---

## 🏎️ Model Performance Comparison

### 1. Performance Metrics
| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|:--------------|:--------:|:---------:|:---------:|:------:|:--------:|:---:|
| **XGBoost (Ensemble)** | **0.8745** | **0.9421** | 0.8771 | 0.8745 | 0.8753 | 0.8122 |
| **Random Forest (Ensemble)** | 0.8471 | 0.9158 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| **Logistic Regression** | 0.8078 | 0.8812 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| **Decision Tree** | 0.8078 | 0.8545 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| **KNN** | 0.8039 | 0.8421 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| **Naive Bayes** | 0.5137 | 0.6542 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

### 2. Technical Observations
- **Ensemble Dominance**: XGBoost and Random Forest significantly outperform traditional models, with XGBoost achieving the highest accuracy (**87.45%**) and AUC (**0.94**).
- **Linear Baseline**: Logistic Regression provides a solid baseline (80.78%), showing that the laptop price data has strong linear relationships with hardware specs.
- **Naive Bayes Limitation**: Performs poorly (51.37%) likely due to the strong correlation between features (e.g., Ram and CPU brand), violating its independence assumption.

---

## 📁 Repository Structure
```
├── app.py                          # Streamlit web application
├── train_models.py                 # Model training & evaluation script
├── requirements.txt                # Python dependencies
├── laptop_data_cleaned.csv         # Cleaned dataset
├── model_comparison_metrics.csv    # Exported evaluation metrics (incl. AUC)
├── all_models.pkl                  # Serialized model pipelines
└── README.md                       # Project documentation
```

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repository
git clone https://github.com/Rakeshraja06/ML-assingment-2-laptop-price-predictor-
cd ML-assingment-2-laptop-price-predictor-

# Install dependencies
pip install -r requirements.txt
```

### Running the App
```bash
streamlit run app.py
```

---

## 🖥️ Streamlit App Features
- **Batch Prediction**: Upload a CSV file and download category predictions.
- **Model Selector**: Switch between all 6 trained models in real-time.
- **Performance Dashboard**: Interactive heatmaps and class-wise performance reports.
- **Metric Cards**: Instant visibility into Accuracy, AUC, F1, and MCC for the selected model.

---

## 🛠️ Technologies Used
- **ML**: Scikit-learn, XGBoost, Joblib
- **Data**: Pandas, NumPy
- **Visualization**: Seaborn, Matplotlib, Plotly
- **Interface**: Streamlit
