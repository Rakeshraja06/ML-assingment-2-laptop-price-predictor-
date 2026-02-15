
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ─── Global ─── */
    .main .block-container { padding-top: 1.5rem; }

    /* ─── Header banner ─── */
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        color: #fff;
        margin-bottom: 1.5rem;
    }
    .hero-banner h1 { margin: 0; font-size: 2.2rem; }
    .hero-banner p  { margin: 0.4rem 0 0 0; opacity: 0.9; font-size: 1.05rem; }

    /* ─── Metric cards ─── */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .metric-label { color: #aaa; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #fff; font-size: 1.7rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-value.good { color: #2ecc71; }
    .metric-value.ok   { color: #f39c12; }
    .metric-value.bad  { color: #e74c3c; }

    /* ─── Model-type badge ─── */
    .badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-ensemble  { background: #27ae6020; color: #2ecc71; border: 1px solid #2ecc7140; }
    .badge-traditional { background: #3498db20; color: #5dade2; border: 1px solid #3498db40; }

    /* ─── Info box ─── */
    .model-info-box {
        background: #1a1a2e;
        border-left: 4px solid #667eea;
        border-radius: 8px;
        padding: 1rem 1.3rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.92rem;
        color: #ccc;
    }

    /* ─── Section divider ─── */
    .section-divider {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }

    /* ─── Prediction result ─── */
    .pred-result {
        text-align: center;
        padding: 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
    }
    .pred-low    { background: linear-gradient(135deg, #27ae6020, #2ecc7110); border: 2px solid #2ecc71; }
    .pred-medium { background: linear-gradient(135deg, #f39c1220, #e67e2210); border: 2px solid #f39c12; }
    .pred-high   { background: linear-gradient(135deg, #e74c3c20, #c0392b10); border: 2px solid #e74c3c; }
    .pred-result h2 { margin: 0; font-size: 1.8rem; }
    .pred-result p  { margin: 0.3rem 0 0 0; opacity: 0.8; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Constants & Model Metadata
# ──────────────────────────────────────────────
ALL_MODELS_PATH = "all_models.pkl"

MODEL_INFO = {
    "XGBoost": {
        "type": "Ensemble",
        "technique": "Gradient Boosting",
        "icon": "🚀",
        "desc": "XGBoost uses **gradient boosting** — it trains decision trees sequentially, where each new tree corrects the errors of the previous ones. This makes it one of the most powerful ML algorithms.",
    },
    "Random Forest": {
        "type": "Ensemble",
        "technique": "Bagging",
        "icon": "🌲",
        "desc": "Random Forest uses **bagging (Bootstrap Aggregation)** — it trains many decision trees on random subsets of data and averages their predictions to reduce overfitting.",
    },
    "Logistic Regression": {
        "type": "Traditional",
        "technique": "Linear Model",
        "icon": "📈",
        "desc": "Logistic Regression is a **linear classifier** that models the probability of each class using a logistic function. Simple, fast, and highly interpretable.",
    },
    "Decision Tree": {
        "type": "Traditional",
        "technique": "Tree-Based",
        "icon": "🌳",
        "desc": "Decision Tree splits data recursively based on feature thresholds. Easy to understand but can overfit without pruning.",
    },
    "KNN": {
        "type": "Traditional",
        "technique": "Instance-Based",
        "icon": "📍",
        "desc": "K-Nearest Neighbours classifies by finding the K closest training samples and taking a majority vote. No explicit training phase.",
    },
    "Naive Bayes": {
        "type": "Traditional",
        "technique": "Probabilistic",
        "icon": "🎲",
        "desc": "Naive Bayes applies **Bayes' theorem** with the assumption of feature independence. Very fast, works well on small datasets.",
    },
}

# ──────────────────────────────────────────────
# Load models
# ──────────────────────────────────────────────
@st.cache_resource
def load_all_models():
    if os.path.exists(ALL_MODELS_PATH):
        return joblib.load(ALL_MODELS_PATH)
    return None

def get_model(name):
    models = load_all_models()
    if models and name in models:
        return models[name]
    return None

@st.cache_data
def load_metrics():
    if os.path.exists("model_comparison_metrics.csv"):
        return pd.read_csv("model_comparison_metrics.csv")
    return None

# ──────────────────────────────────────────────
# Helper: colour score
# ──────────────────────────────────────────────
def score_class(v):
    if v >= 0.80: return "good"
    if v >= 0.60: return "ok"
    return "bad"

def metric_card(label, value):
    cls = score_class(value)
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {cls}">{value:.2%}</div>
    </div>"""

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # Category filter
    category = st.radio("Model Category", ["All", "🟢 Ensemble", "🔵 Traditional"], horizontal=False)
    
    if category == "🟢 Ensemble":
        model_list = [m for m in MODEL_INFO if MODEL_INFO[m]["type"] == "Ensemble"]
    elif category == "🔵 Traditional":
        model_list = [m for m in MODEL_INFO if MODEL_INFO[m]["type"] == "Traditional"]
    else:
        model_list = list(MODEL_INFO.keys())

    selected_model_name = st.selectbox("Select Model", model_list)

    info = MODEL_INFO[selected_model_name]
    badge_cls = "badge-ensemble" if info["type"] == "Ensemble" else "badge-traditional"
    st.markdown(f'{info["icon"]} <span class="badge {badge_cls}">{info["type"]} · {info["technique"]}</span>', unsafe_allow_html=True)
    st.markdown(info["desc"])

    st.markdown("---")
    # Quick metrics in sidebar
    metrics_df = load_metrics()
    if metrics_df is not None:
        row = metrics_df[metrics_df["ML Model Name"] == selected_model_name]
        if not row.empty:
            acc = row["Accuracy"].values[0]
            f1  = row["F1 Score"].values[0]
            st.metric("Accuracy", f"{acc:.2%}")
            st.metric("F1 Score", f"{f1:.2%}")

# ──────────────────────────────────────────────
# Hero Banner
# ──────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <h1>💻 Laptop Price Category Predictor</h1>
    <p>ML Assignment 2 &nbsp;·&nbsp; <strong>Rakesh R (2024dc04070)</strong></p>
    <p>Predict whether a laptop falls into <b>Low</b>, <b>Medium</b>, or <b>High</b> price category using 6 ML models.</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Load selected model
# ──────────────────────────────────────────────
model = get_model(selected_model_name)

if model is None:
    st.error(f"⚠️ Model **{selected_model_name}** not found! Run `python train_models.py` first.")
    st.stop()

# ══════════════════════════════════════════════
# TAB LAYOUT
# ══════════════════════════════════════════════
tab_perf, tab_predict, tab_batch = st.tabs([
    "📊 Model Performance",
    "🔮 Predict",
    "📁 Batch Prediction",
])

# ──────────────────────────────────────────────
# TAB 1 — Model Performance
# ──────────────────────────────────────────────
with tab_perf:
    if metrics_df is not None:
        model_row = metrics_df[metrics_df["ML Model Name"] == selected_model_name]
        metric_cols = ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]

        # ── Model info banner ──
        info = MODEL_INFO[selected_model_name]
        badge_cls = "badge-ensemble" if info["type"] == "Ensemble" else "badge-traditional"
        st.markdown(f"""
        <div class="model-info-box">
            {info["icon"]} <span class="badge {badge_cls}">{info["type"]} · {info["technique"]}</span>
            &nbsp;—&nbsp; {info["desc"]}
        </div>
        """, unsafe_allow_html=True)

        # ── KPI metric cards ──
        if not model_row.empty:
            r = model_row.iloc[0]
            cols = st.columns(5)
            for col, (label, key) in zip(cols, [
                ("Accuracy", "Accuracy"),
                ("Precision", "Precision"),
                ("Recall", "Recall"),
                ("F1 Score", "F1 Score"),
                ("MCC", "MCC"),
            ]):
                col.markdown(metric_card(label, r[key]), unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── 🏆 Best Model Highlight ──
        best_model_row = metrics_df.loc[metrics_df["Accuracy"].idxmax()]
        best_name = best_model_row["ML Model Name"]
        best_info = MODEL_INFO[best_name]
        best_badge = "badge-ensemble" if best_info["type"] == "Ensemble" else "badge-traditional"

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #1a472a 0%, #2d4a3e 100%);
                    border: 1px solid #2ecc7140; border-radius: 14px; padding: 1.2rem 1.8rem; margin-bottom: 1.2rem;">
            <span style="font-size:1.4rem;">🏆</span>
            <span style="color:#2ecc71; font-size:1.1rem; font-weight:700;"> Best Model: {best_name}</span>
            &nbsp; <span class="badge {best_badge}">{best_info["type"]}</span>
            <span style="color:#aaa; margin-left:1rem;">
                Accuracy: <b style="color:#2ecc71">{best_model_row["Accuracy"]:.2%}</b> &nbsp;|&nbsp;
                F1: <b style="color:#2ecc71">{best_model_row["F1 Score"]:.2%}</b> &nbsp;|&nbsp;
                MCC: <b style="color:#2ecc71">{best_model_row["MCC"]:.2%}</b>
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ── 🥇 Model Ranking Cards ──
        st.subheader("🥇 Model Rankings (by Accuracy)")
        ranked = metrics_df.sort_values("Accuracy", ascending=False).reset_index(drop=True)
        rank_emojis = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"]
        rank_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#667eea", "#764ba2", "#e74c3c"]

        rank_cols = st.columns(len(ranked))
        for i, (col, (_, row)) in enumerate(zip(rank_cols, ranked.iterrows())):
            m_info = MODEL_INFO[row["ML Model Name"]]
            is_selected = row["ML Model Name"] == selected_model_name
            border_style = f"border: 2px solid {rank_colors[i]};" if is_selected else f"border: 1px solid rgba(255,255,255,0.08);"
            col.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e1e2f, #2d2d44);
                        {border_style} border-radius: 12px; padding: 0.8rem; text-align:center; min-height:140px;">
                <div style="font-size:1.5rem;">{rank_emojis[i]}</div>
                <div style="color:#fff; font-weight:600; font-size:0.85rem; margin:0.3rem 0;">{m_info["icon"]} {row["ML Model Name"]}</div>
                <div style="color:{rank_colors[i]}; font-size:1.3rem; font-weight:700;">{row["Accuracy"]:.1%}</div>
                <div style="color:#aaa; font-size:0.72rem;">F1: {row["F1 Score"]:.1%} · MCC: {row["MCC"]:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Heatmap: metric scores ──
        st.subheader("🌡️ Performance Heatmap")
        heatmap_df = metrics_df.set_index("ML Model Name")[metric_cols]
        fig_heat = go.Figure(go.Heatmap(
            z=heatmap_df.values,
            x=metric_cols,
            y=heatmap_df.index.tolist(),
            colorscale=[[0, "#e74c3c"], [0.5, "#f39c12"], [0.75, "#f1c40f"], [1, "#2ecc71"]],
            text=[[f"{v:.2%}" for v in row] for row in heatmap_df.values],
            texttemplate="%{text}",
            textfont=dict(size=13, color="#fff"),
            zmin=0.3, zmax=1.0,
            colorbar=dict(title="Score", tickformat=".0%"),
        ))
        fig_heat.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

        # ── Bar chart: all models comparison ──
        st.subheader("📊 All Models — Metric Comparison")
        melted = metrics_df.melt(
            id_vars="ML Model Name",
            value_vars=metric_cols,
            var_name="Metric",
            value_name="Score"
        )
        fig_bar = px.bar(
            melted, x="ML Model Name", y="Score", color="Metric",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"ML Model Name": "Model", "Score": "Score"},
            height=420,
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#ccc",
            legend=dict(orientation="h", y=-0.18),
            yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Radar chart + Comparison Table side by side ──
        col_radar, col_table = st.columns([1, 1])

        with col_radar:
            st.subheader(f"🕸️ {selected_model_name} — Performance Profile")
            if not model_row.empty:
                r = model_row.iloc[0]
                values = [r[m] for m in metric_cols]
                values.append(values[0])  # close the radar
                labels = metric_cols + [metric_cols[0]]

                fig_radar = go.Figure(go.Scatterpolar(
                    r=values,
                    theta=labels,
                    fill="toself",
                    fillcolor="rgba(102,126,234,0.25)",
                    line=dict(color="#667eea", width=2),
                    marker=dict(size=6),
                ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.08)"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    height=380,
                    margin=dict(t=30, b=30),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        with col_table:
            st.subheader("📋 Full Comparison Table")
            # Color-coded table: green for best, red-ish for worst
            def color_cells(val, col_min, col_max):
                if col_max == col_min:
                    return ""
                ratio = (val - col_min) / (col_max - col_min)
                if ratio >= 0.8:
                    return "background-color: rgba(46,204,113,0.25); color: #2ecc71; font-weight:700"
                elif ratio >= 0.6:
                    return "background-color: rgba(241,196,15,0.15); color: #f1c40f"
                elif ratio >= 0.4:
                    return "background-color: rgba(243,156,18,0.15); color: #f39c12"
                else:
                    return "background-color: rgba(231,76,60,0.15); color: #e74c3c"

            def style_metrics(df):
                styles = pd.DataFrame("", index=df.index, columns=df.columns)
                for col in metric_cols:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    styles[col] = df[col].apply(lambda v: color_cells(v, col_min, col_max))
                return styles

            styled = metrics_df.style.format(precision=4).apply(
                lambda _: style_metrics(metrics_df), axis=None
            )
            st.dataframe(styled, hide_index=True, use_container_width=True, height=380)

        # ── Feature importance (ensemble models only) ──
        if info["type"] == "Ensemble":
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            st.subheader(f"🔍 Feature Importance — {selected_model_name}")

            try:
                clf = model.named_steps["classifier"]
                preprocessor = model.named_steps["preprocessor"]

                # Get feature names
                feature_names = preprocessor.get_feature_names_out()
                # Clean up prefixes
                feature_names = [f.replace("num__", "").replace("cat__", "") for f in feature_names]

                importances = clf.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values("Importance", ascending=True).tail(15)

                fig_fi = px.bar(
                    fi_df, x="Importance", y="Feature", orientation="h",
                    color="Importance",
                    color_continuous_scale=["#764ba2", "#667eea", "#2ecc71"],
                    height=450,
                )
                fig_fi.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    coloraxis_showscale=False,
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                    margin=dict(l=10),
                )
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not extract feature importance: {e}")
    else:
        st.error("model_comparison_metrics.csv not found!")

# ──────────────────────────────────────────────
# TAB 2 — Single Prediction
# ──────────────────────────────────────────────
with tab_predict:
    st.subheader("🔮 Predict Laptop Price Category")
    st.caption(f"Using **{selected_model_name}**")

    # Categorical options
    brands = ['Apple', 'HP', 'Acer', 'Asus', 'Dell', 'Lenovo', 'Chuwi', 'MSI',
              'Microsoft', 'Toshiba', 'Huawei', 'Xiaomi', 'Vero', 'Razer',
              'Mediacom', 'Samsung', 'Google', 'Fujitsu', 'LG']
    types = ['Ultrabook', 'Notebook', 'Netbook', 'Gaming', '2 in 1 Convertible', 'Workstation']
    cpu_brands = ['Intel Core i5', 'Intel Core i7', 'AMD', 'Intel Core i3', 'Other Intel Processor']
    gpu_brands = ['Intel', 'AMD', 'Nvidia']
    os_list = ['Mac', 'Windows', 'No OS', 'Linux', 'Android', 'Chrome OS', 'Others']

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🏢 Brand & Type")
        company = st.selectbox("Brand", brands, key="brand")
        type_name = st.selectbox("Type", types, key="type")
        ram = st.slider("RAM (GB)", 2, 64, 8, key="ram")
        weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.5, 0.01, key="weight")

    with col2:
        st.markdown("##### 🖥️ Display & CPU")
        touchscreen = st.selectbox("Touchscreen", ["No", "Yes"], key="touch")
        ips = st.selectbox("IPS Panel", ["No", "Yes"], key="ips")
        ppi = st.number_input("PPI", 50, 400, 141, key="ppi")
        cpu_brand = st.selectbox("CPU Brand", cpu_brands, key="cpu")

    with col3:
        st.markdown("##### 💾 Storage & GPU")
        hdd = st.selectbox("HDD (GB)", [0, 128, 256, 500, 1000, 2000], key="hdd")
        ssd = st.selectbox("SSD (GB)", [0, 8, 16, 32, 64, 128, 256, 512, 1000], key="ssd")
        gpu_brand = st.selectbox("GPU Brand", gpu_brands, key="gpu")
        os_sys = st.selectbox("Operating System", os_list, key="os")

    st.markdown("")
    if st.button("⚡ Predict Price Category", type="primary", use_container_width=True):
        input_df = pd.DataFrame({
            'Company': [company], 'TypeName': [type_name], 'Ram': [ram], 'Weight': [weight],
            'TouchScreen': [1 if touchscreen == "Yes" else 0],
            'Ips': [1 if ips == "Yes" else 0],
            'Ppi': [ppi], 'Cpu_brand': [cpu_brand],
            'HDD': [hdd], 'SSD': [ssd], 'Gpu_brand': [gpu_brand], 'Os': [os_sys]
        })

        try:
            prediction = model.predict(input_df)[0]
            categories = ["Low", "Medium", "High"]

            if isinstance(prediction, (int, np.integer)):
                predicted_label = categories[prediction]
            else:
                predicted_label = prediction

            # Colour-coded result
            style_map = {"Low": "pred-low", "Medium": "pred-medium", "High": "pred-high"}
            emoji_map = {"Low": "💚", "Medium": "🧡", "High": "❤️"}
            cls = style_map.get(predicted_label, "pred-medium")
            emo = emoji_map.get(predicted_label, "")
            st.markdown(f"""
            <div class="pred-result {cls}">
                <h2>{emo} {predicted_label}</h2>
                <p>Predicted Price Category</p>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)[0]
                if hasattr(model, "classes_"):
                    labels = [str(c) for c in model.classes_]
                    if all(c.isdigit() for c in labels):
                        labels = categories
                else:
                    labels = categories

                prob_df = pd.DataFrame({"Category": labels, "Probability": proba})
                fig_prob = px.bar(
                    prob_df, x="Probability", y="Category", orientation="h",
                    color="Category",
                    color_discrete_map={
                        "Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c",
                        "0": "#2ecc71", "1": "#f39c12", "2": "#e74c3c",
                    },
                    text=prob_df["Probability"].apply(lambda x: f"{x:.1%}"),
                    height=220,
                )
                fig_prob.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font_color="#ccc",
                    showlegend=False,
                    xaxis=dict(range=[0, 1], gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                fig_prob.update_traces(textposition="outside")
                st.plotly_chart(fig_prob, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ──────────────────────────────────────────────
# TAB 3 — Batch Prediction
# ──────────────────────────────────────────────
with tab_batch:
    st.subheader("📁 Batch Prediction via CSV")
    st.caption(f"Using **{selected_model_name}**")

    required_cols = ['Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen',
                     'Ips', 'Ppi', 'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'Os']

    with st.expander("📋 Required CSV Schema", expanded=False):
        schema_info = pd.DataFrame({
            'Column': required_cols,
            'Type': ['Categorical', 'Categorical', 'Numerical', 'Numerical',
                     'Binary (0/1)', 'Binary (0/1)', 'Numerical', 'Categorical',
                     'Numerical', 'Numerical', 'Categorical', 'Categorical'],
            'Example': ['Apple', 'Ultrabook', '8', '1.37', '0', '1', '227',
                        'Intel Core i5', '0', '128', 'Intel', 'Mac'],
        })
        st.dataframe(schema_info, hide_index=True, use_container_width=True)

    # Template download
    example_data = {
        'Company': ['Apple'], 'TypeName': ['Ultrabook'], 'Ram': [8], 'Weight': [1.37],
        'TouchScreen': [0], 'Ips': [1], 'Ppi': [227], 'Cpu_brand': ['Intel Core i5'],
        'HDD': [0], 'SSD': [128], 'Gpu_brand': ['Intel'], 'Os': ['Mac']
    }
    template_csv = pd.DataFrame(example_data).to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Template CSV", template_csv, "laptop_template.csv", "text/csv")

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

    if uploaded_file:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("**Preview:**")
            st.dataframe(batch_df.head(), use_container_width=True)

            missing = [c for c in required_cols if c not in batch_df.columns]
            if missing:
                st.error(f"❌ Missing columns: {missing}")
            else:
                if st.button("⚡ Run Batch Prediction", type="primary", use_container_width=True):
                    predictions = model.predict(batch_df[required_cols])

                    categories = ["Low", "Medium", "High"]
                    if isinstance(predictions[0], (int, np.integer, float, np.floating)):
                        batch_df["Predicted_Category"] = [categories[int(p)] for p in predictions]
                    else:
                        batch_df["Predicted_Category"] = predictions

                    st.success(f"✅ Predicted **{len(batch_df)}** laptops!")

                    # Summary chart
                    counts = batch_df["Predicted_Category"].value_counts().reindex(categories, fill_value=0)
                    fig_summary = px.pie(
                        names=counts.index, values=counts.values,
                        color=counts.index,
                        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                        hole=0.45, height=300,
                    )
                    fig_summary.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#ccc",
                        margin=dict(t=20, b=20),
                    )
                    col_chart, col_data = st.columns([1, 2])
                    with col_chart:
                        st.plotly_chart(fig_summary, use_container_width=True)
                    with col_data:
                        st.dataframe(batch_df, use_container_width=True)

                    csv_out = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Results", csv_out, "laptop_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")
