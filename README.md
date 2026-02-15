# 💻 ML Assignment 2 — Laptop Price Classification

> **Multi-class classification** of laptops into **Low**, **Medium**, and **High** price categories using six machine learning models, including **ensemble methods** (Random Forest & XGBoost).

[![Open in Streamlit]([https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://jnqrrbkmfgg45ywjtnoyrb.streamlit.app/)](https://share.streamlit.io)

---

## 📋 Problem Statement

Build and compare **six classification algorithms** — including both traditional and ensemble methods — on a real-world laptop dataset to predict price categories derived from continuous prices via quantile-based binning.

---

## 🏆 Model Performance Comparison

| Rank | Model | Type | Accuracy | Precision | Recall | F1 Score | MCC |
|:----:|-------|:----:|:--------:|:---------:|:------:|:--------:|:---:|
| 🥇 | **XGBoost** | Ensemble | **0.8745** | **0.8771** | **0.8745** | **0.8753** | **0.8122** |
| 🥈 | Random Forest | Ensemble | 0.8471 | 0.8538 | 0.8471 | 0.8491 | 0.7717 |
| 🥉 | Logistic Regression | Traditional | 0.8078 | 0.8237 | 0.8078 | 0.8116 | 0.7159 |
| 4 | Decision Tree | Traditional | 0.8078 | 0.8160 | 0.8078 | 0.8100 | 0.7133 |
| 5 | KNN | Traditional | 0.8039 | 0.8117 | 0.8039 | 0.8061 | 0.7073 |
| 6 | Naive Bayes | Traditional | 0.5137 | 0.6031 | 0.5137 | 0.4731 | 0.3064 |

> **🚀 Key Result:** The **XGBoost** ensemble model achieved the highest accuracy of **87.45%**, significantly outperforming traditional methods like Naive Bayes (51.37%) and basic Decision Trees (80.78%).

---

## 📊 Dataset Description

| Property | Detail |
|----------|--------|
| **Source** | Laptop Price Dataset (cleaned) |
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

**Target variable** is created by splitting continuous prices into three equal-frequency bins using `pd.qcut`:
- **Low** — Bottom 33 % (425 instances)
- **Medium** — Middle 33 % (424 instances)
- **High** — Top 33 % (424 instances)

</details>

---

## 🤖 Models Used

### 🟢 Ensemble Models
| Model | Technique | Description |
|-------|-----------|-------------|
| **Random Forest** | Bagging | Trains 100 decision trees on random data subsets; aggregates predictions to reduce overfitting |
| **XGBoost** | Gradient Boosting | Trains trees sequentially — each correcting errors of the previous; state-of-the-art performance |

### 🔵 Traditional Models
| Model | Technique | Description |
|-------|-----------|-------------|
| **Logistic Regression** | Linear Model | Models class probabilities using a logistic function; strong interpretable baseline |
| **Decision Tree** | Tree-Based | Splits data recursively on feature thresholds; easy to interpret but prone to overfitting |
| **KNN** | Instance-Based | Classifies by majority vote of K nearest neighbours; no explicit training phase |
| **Naive Bayes** | Probabilistic | Applies Bayes' theorem assuming feature independence; very fast on small data |

---

---

## 📁 Repository Structure

```
ML Assingment 2/
│
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
├── laptop_classification.ipynb     # Full EDA + training notebook
├── train_models.py                 # Standalone training script
├── app.py                          # Streamlit web application
├── laptop_data_cleaned.csv         # Cleaned dataset
├── model_comparison_metrics.csv    # Evaluation results
├── all_models.pkl                  # All 6 trained model pipelines (single file)
│
└── archive/                        # Previous code versions
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/ML-Assignment-2.git
cd ML-Assignment-2

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate    # macOS/Linux
# .venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Train Models (optional — pre-trained `all_models.pkl` is included)

```bash
python train_models.py
```

### Run the Streamlit App Locally

```bash
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repository to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select your GitHub repo
4. Set:
   - **Main file path:** `app.py`
   - **Branch:** `main`
5. Click **Deploy!**

> The app will automatically install dependencies from `requirements.txt` and launch.

---

## 🖥️ Streamlit App Features

| Feature | Description |
|---------|-------------|
| 🎨 **Rich UI** | Gradient banners, dark-themed metric cards, color-coded performance indicators |
| 📊 **Interactive Charts** | Plotly bar charts, radar charts, heatmaps for model comparison |
| 🏆 **Model Rankings** | Visual leaderboard with medal emojis and accuracy rankings |
| 🔍 **Feature Importance** | Interactive charts for ensemble models (Random Forest, XGBoost) |
| 🕸️ **Radar Profile** | Performance profile of selected model across all metrics |
| ⚙️ **Model Selector** | Filter by Ensemble/Traditional, select any of 6 models |
| 🔮 **Manual Prediction** | Enter laptop specs and get instant price category prediction |
| 📁 **Batch Prediction** | Upload CSV (template provided) for bulk predictions with donut chart summary |
| 📥 **Download Results** | Download batch prediction results as CSV |

---

## 🛠️ Technologies Used

| Category | Tool |
|----------|------|
| Language | Python 3.8+ |
| ML Framework | scikit-learn, XGBoost |
| Data | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Web App | Streamlit |
| Serialization | Joblib |
| Deployment | Streamlit Cloud |



## 👤 Author (student id: 2024dc04070)

**Rakesh R** (2024dc04070)  
BITS Pilani — Machine Learning, Assignment 2

---

## 📄 License

This project is created for educational purposes as part of BITS Pilani coursework.

---

**Submission Date:** February 15, 2026  
**Course:** Machine Learning  
**Institution:** BITS Pilani
