"""
Customer Churn Prediction — Interactive Dashboard
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ============================================================
# Page config & Custom CSS
# ============================================================

st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Font & base */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide streamlit default header/footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.stApp {
    background-color: #0f1117;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #161b27;
    border-right: 1px solid #1e2535;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #1a2035;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 500;
    color: #8892a4 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
[data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background-color: #161b27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2535;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: 500;
    font-size: 13px;
    color: #8892a4;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background-color: #1e3a5f !important;
    color: #60a5fa !important;
}

/* Section headers */
h1 { color: #e2e8f0 !important; font-weight: 700 !important; font-size: 24px !important; }
h2 { color: #e2e8f0 !important; font-weight: 600 !important; font-size: 20px !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; font-size: 16px !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #1e2535;
    border-radius: 10px;
    overflow: hidden;
}

/* Info / success / error boxes */
.stAlert {
    border-radius: 10px;
    border: none;
}

/* Slider */
[data-testid="stSlider"] [data-baseweb="slider"] {
    margin-top: 4px;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background-color: #1a2035;
    border-color: #1e2535;
    border-radius: 8px;
}

/* Custom card container */
.card {
    background-color: #1a2035;
    border: 1px solid #1e2535;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
}

/* Section divider */
.section-label {
    font-size: 11px;
    font-weight: 600;
    color: #60a5fa;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
}

/* Sidebar model info */
.sidebar-info {
    background-color: #1e2d4a;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 8px;
    border: 1px solid #2a3f5f;
}
.sidebar-info-label {
    font-size: 11px;
    color: #8892a4;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.sidebar-info-value {
    font-size: 18px;
    font-weight: 700;
    color: #60a5fa;
    margin-top: 2px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# Matplotlib dark theme
# ============================================================

plt.rcParams.update({
    "figure.facecolor":  "#1a2035",
    "axes.facecolor":    "#1a2035",
    "axes.edgecolor":    "#2d3748",
    "axes.labelcolor":   "#a0aec0",
    "axes.titlecolor":   "#e2e8f0",
    "xtick.color":       "#718096",
    "ytick.color":       "#718096",
    "text.color":        "#e2e8f0",
    "grid.color":        "#2d3748",
    "grid.linewidth":    0.5,
    "font.family":       "sans-serif",
    "font.size":         10,
})

ACCENT   = "#60a5fa"
RED      = "#f87171"
GREEN    = "#34d399"
ORANGE   = "#fb923c"
BG_CARD  = "#1a2035"

# ============================================================
# Load data & model
# ============================================================

@st.cache_data
def load_data():
    X_test  = pd.read_csv("data/X_test.csv")
    X_train = pd.read_csv("data/X_train.csv")
    y_test  = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_test

@st.cache_resource
def load_model_and_meta():
    with open("data/model_metadata.json", "r") as f:
        meta = json.load(f)
    model = joblib.load("data/model_final.pkl")
    return model, meta

X_train, X_test, y_test = load_data()
model, metadata          = load_model_and_meta()
MODEL_NAME        = metadata["model_name"]
DEFAULT_THRESHOLD = metadata["threshold"]
FEATURE_COLS      = X_test.columns.tolist()
y_proba_default   = model.predict_proba(X_test)[:, 1]

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📊 Churn Prediction")
    st.markdown("---")

    st.markdown(f"""
    <div class="sidebar-info">
        <div class="sidebar-info-label">Model</div>
        <div class="sidebar-info-value" style="font-size:14px">{MODEL_NAME}</div>
    </div>
    <div class="sidebar-info">
        <div class="sidebar-info-label">ROC-AUC</div>
        <div class="sidebar-info-value">{metadata['roc_auc']}</div>
    </div>
    <div class="sidebar-info">
        <div class="sidebar-info-label">Recall (optimal)</div>
        <div class="sidebar-info-value">{metadata['recall']:.1%}</div>
    </div>
    <div class="sidebar-info">
        <div class="sidebar-info-label">Optimal Threshold</div>
        <div class="sidebar-info-value">{DEFAULT_THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.10, max_value=0.70,
        value=float(DEFAULT_THRESHOLD), step=0.05,
        help="Turunkan threshold untuk meningkatkan recall (menangkap lebih banyak churn)."
    )
    st.caption(f"Threshold optimal dari training: **{DEFAULT_THRESHOLD}** (F2-score tertinggi)")

y_pred = (y_proba_default >= threshold).astype(int)

# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Model Performance",
    "💰  Risk & Business Impact",
    "🔍  Interpretation",
    "🧑  Predict Customer"
])

# ============================================================
# TAB 1 — Model Performance
# ============================================================

with tab1:
    st.markdown("### Model Performance")
    st.markdown(f"<div class='section-label'>Threshold = {threshold:.2f}</div>", unsafe_allow_html=True)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall", f"{recall:.1%}", help="Dari semua yang benar-benar churn, berapa persen terdeteksi.")
    c2.metric("Precision", f"{precision:.1%}", help="Dari semua yang diprediksi churn, berapa persen benar.")
    c3.metric("ROC-AUC", f"{metadata['roc_auc']}")
    c4.metric("F1-Score", f"{f1:.3f}")

    st.markdown(f"""
    <div class="card">
        Pada threshold <b>{threshold:.2f}</b>: model mendeteksi
        <b style="color:{GREEN}">{tp} dari {tp+fn}</b> customer churn (recall {recall:.1%}),
        dengan <b style="color:{ORANGE}">{fp} false alarm</b> dan
        <b style="color:{RED}">{fn} customer churn terlewat</b>.
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        cmap = mpl.colors.LinearSegmentedColormap.from_list("blue_dark", ["#1a2035", "#1e3a5f", "#60a5fa"])
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax, linewidths=0.5,
                    linecolor="#2d3748",
                    xticklabels=["No Churn", "Churn"],
                    yticklabels=["No Churn", "Churn"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_xlabel("Predicted", labelpad=10)
        ax.set_ylabel("Actual", labelpad=10)
        ax.set_title(f"Confusion Matrix  |  threshold={threshold:.2f}", pad=12)
        fig.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.markdown("#### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba_default)
        roc_auc     = auc(fpr, tpr)
        fig2, ax2   = plt.subplots(figsize=(4.5, 3.5))
        ax2.plot(fpr, tpr, color=ACCENT, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        ax2.fill_between(fpr, tpr, alpha=0.08, color=ACCENT)
        ax2.plot([0,1],[0,1], "--", color="#4a5568", linewidth=1, label="Random")
        ax2.set_xlabel("False Positive Rate", labelpad=10)
        ax2.set_ylabel("True Positive Rate", labelpad=10)
        ax2.set_title("ROC Curve", pad=12)
        ax2.legend(framealpha=0.2, edgecolor="#2d3748")
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Threshold Trade-off")
    rows = []
    for t in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        pt = (y_proba_default >= t).astype(int)
        cmt = confusion_matrix(y_test, pt)
        tnt, fpt, fnt, tpt = cmt.ravel()
        rec  = tpt / (tpt + fnt) if (tpt + fnt) > 0 else 0
        prec = tpt / (tpt + fpt) if (tpt + fpt) > 0 else 0
        rows.append({
            "Threshold": t,
            "Recall": f"{rec:.1%}",
            "Precision": f"{prec:.1%}",
            "FN (terlewat)": fnt,
            "FP (false alarm)": fpt,
            "": "✅ Dipilih" if t == DEFAULT_THRESHOLD else ""
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 — Risk Segmentation & Business Impact
# ============================================================

with tab2:
    st.markdown("### Risk Segmentation & Business Impact")

    risk_df = pd.DataFrame({
        "actual_churn":    y_test.values,
        "churn_proba":     y_proba_default,
        "monthly_charges": X_test["MonthlyCharges"].values
    })

    c1, c2 = st.columns(2)
    with c1:
        high_cut = st.slider("High Risk threshold (≥)", 0.5, 0.9, 0.6, 0.05)
    with c2:
        med_cut = st.slider("Medium Risk threshold (≥)", 0.2,
                             float(high_cut)-0.05, DEFAULT_THRESHOLD, 0.05)

    def segment(p):
        if p >= high_cut:  return "High Risk"
        elif p >= med_cut: return "Medium Risk"
        else:              return "Low Risk"

    risk_df["segment"] = risk_df["churn_proba"].apply(segment)
    summary = risk_df.groupby("segment").agg(
        Jumlah_Customer     = ("actual_churn", "count"),
        Actual_Churn_Rate   = ("actual_churn", "mean"),
        Avg_Monthly_Charges = ("monthly_charges", "mean")
    ).reindex(["High Risk", "Medium Risk", "Low Risk"]).dropna()

    col_l, col_r = st.columns([1.1, 1])

    with col_l:
        st.markdown("#### Segment Summary")
        st.dataframe(
            summary.style.format({
                "Actual_Churn_Rate":   "{:.1%}",
                "Avg_Monthly_Charges": "${:.2f}"
            }),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(5, 3))
        seg_colors = [RED, ORANGE, GREEN]
        bars = ax.bar(summary.index, summary["Actual_Churn_Rate"],
                      color=seg_colors, width=0.5, edgecolor="none")
        ax.set_ylabel("Actual Churn Rate", labelpad=10)
        ax.set_ylim(0, 1)
        ax.set_title("Churn Rate per Segment", pad=12)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        for bar, val in zip(bars, summary["Actual_Churn_Rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.1%}", ha="center", fontweight="700",
                    fontsize=12, color="#e2e8f0")
        ax.spines[["top","right","left"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.markdown("#### 💰 Business Impact Calculator")
        success_rate = st.slider(
            "Retention success rate", 0.0, 1.0, 0.30, 0.05,
            help="% customer High Risk yang benar-benar churn namun berhasil diretensi."
        )

        high_risk       = risk_df[risk_df["segment"] == "High Risk"]
        high_risk_churn = high_risk[high_risk["actual_churn"] == 1]
        n_saved      = len(high_risk_churn) * success_rate
        monthly_rev  = n_saved * high_risk_churn["monthly_charges"].mean() if len(high_risk_churn) > 0 else 0
        annual_rev   = monthly_rev * 12

        m1, m2 = st.columns(2)
        m1.metric("High Risk Customer", f"{len(high_risk)}")
        m2.metric("Benar-benar Churn", f"{len(high_risk_churn)}")
        m1.metric("Estimasi Terselamatkan", f"{n_saved:.0f}")
        m2.metric("Revenue / Tahun", f"${annual_rev:,.0f}")

        st.markdown(f"""
        <div class="card" style="margin-top:12px">
            <div class="section-label">Cara membaca</div>
            Dengan menarget <b style="color:{RED}">{len(high_risk)} customer High Risk</b>
            (churn rate aktual {summary.loc['High Risk','Actual_Churn_Rate']:.1%}),
            campaign retensi dengan success rate {success_rate:.0%} berpotensi
            menyelamatkan <b style="color:{GREEN}">${annual_rev:,.0f}/tahun</b>.
        </div>
        """, unsafe_allow_html=True)
        st.caption("Angka ilustratif — success rate aktual tergantung efektivitas campaign.")

# ============================================================
# TAB 3 — Model Interpretation
# ============================================================

with tab3:
    st.markdown("### Model Interpretation")
    st.markdown(f"""
    <div class="card">
        <div class="section-label">Metode Interpretasi</div>
        Untuk <b>Logistic Regression</b>, interpretasi dilakukan melalui
        <b>koefisien model</b> dan <b>odds ratio</b> — lebih tepat dibanding
        SHAP TreeExplainer (khusus tree-based models).
        Odds ratio menunjukkan <i>seberapa besar</i> suatu fitur meningkatkan
        atau menurunkan risiko churn dibanding baseline.
    </div>
    """, unsafe_allow_html=True)

    coef       = pd.Series(model.coef_[0], index=FEATURE_COLS)
    top_idx    = coef.abs().sort_values(ascending=False).head(15).index
    top_coef   = coef[top_idx].sort_values()
    odds_ratio = np.exp(coef[top_idx]).sort_values(ascending=False)

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Koefisien (Log-Odds)")
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        colors  = [RED if v > 0 else ACCENT for v in top_coef.values]
        bars    = ax.barh(range(len(top_coef)), top_coef.values,
                          color=colors, height=0.6, edgecolor="none")
        ax.set_yticks(range(len(top_coef)))
        ax.set_yticklabels(top_coef.index, fontsize=9)
        ax.axvline(0, color="#4a5568", linewidth=1)
        ax.set_xlabel("Log-Odds Coefficient", labelpad=10)
        ax.set_title("Merah = dorong Churn  |  Biru = dorong No Churn", pad=12, fontsize=10)
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        st.pyplot(fig)

    with col_r:
        st.markdown("#### Odds Ratio")
        or_sorted = odds_ratio.sort_values()
        fig2, ax2 = plt.subplots(figsize=(5.5, 5.5))
        colors2   = [RED if v > 1 else ACCENT for v in or_sorted.values]
        ax2.barh(range(len(or_sorted)), or_sorted.values,
                 color=colors2, height=0.6, edgecolor="none")
        ax2.set_yticks(range(len(or_sorted)))
        ax2.set_yticklabels(or_sorted.index, fontsize=9)
        ax2.axvline(1, color="#4a5568", linewidth=1, linestyle="--")
        ax2.set_xlabel("Odds Ratio  exp(coef)", labelpad=10)
        ax2.set_title(">1 = meningkatkan churn  |  <1 = menurunkan churn", pad=12, fontsize=10)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.grid(axis="x", alpha=0.3)
        fig2.tight_layout()
        st.pyplot(fig2)

    st.markdown("#### Tabel Odds Ratio")
    or_table = pd.DataFrame({
        "Feature":     odds_ratio.index,
        "Coefficient": coef[odds_ratio.index].round(4).values,
        "Odds Ratio":  odds_ratio.round(3).values,
        "Arah":        ["⬆ Meningkatkan churn" if v > 1 else "⬇ Menurunkan churn"
                        for v in odds_ratio.values]
    })
    st.dataframe(or_table, use_container_width=True, hide_index=True)

    with st.expander("📖 Cara membaca Odds Ratio"):
        st.markdown(f"""
        - **OR > 1** → fitur ini **meningkatkan** risiko churn.
          Contoh: OR = 2.5 → customer dengan fitur ini **2.5× lebih mungkin churn**.
        - **OR < 1** → fitur ini **menurunkan** risiko churn.
          Contoh: OR = 0.2 → customer dengan fitur ini hanya **0.2× kemungkinan churn** (lebih aman).
        - **OR = 1** → fitur tidak berpengaruh pada probabilitas churn.
        """)

# ============================================================
# TAB 4 — Predict New Customer
# ============================================================

CATEGORICAL_OPTIONS = {
    "MultipleLines":    ["No", "Yes", "No phone service"],
    "InternetService":  ["DSL", "Fiber optic", "No"],
    "OnlineSecurity":   ["No", "Yes", "No internet service"],
    "OnlineBackup":     ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport":      ["No", "Yes", "No internet service"],
    "StreamingTV":      ["No", "Yes", "No internet service"],
    "StreamingMovies":  ["No", "Yes", "No internet service"],
    "Contract":         ["Month-to-month", "One year", "Two year"],
    "PaymentMethod":    ["Electronic check", "Mailed check",
                         "Bank transfer (automatic)", "Credit card (automatic)"],
}
ONE_HOT_COLS = list(CATEGORICAL_OPTIONS.keys()) + ["tenure_group"]
SERVICE_COLS = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]

with tab4:
    st.markdown("### Predict New Customer")
    st.markdown("""
    <div class="card">
        <div class="section-label">Cara Pakai</div>
        Isi profil customer di bawah, lalu klik <b>Prediksi Churn</b>.
        Model akan menghitung probabilitas churn dan menampilkan faktor-faktor
        yang paling mempengaruhi prediksi untuk customer tersebut secara spesifik.
    </div>
    """, unsafe_allow_html=True)

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("<div class='section-label'>Demografi</div>", unsafe_allow_html=True)
            gender     = st.selectbox("Gender", ["Female","Male"])
            senior     = st.selectbox("Senior Citizen", ["No","Yes"])
            partner    = st.selectbox("Has Partner", ["No","Yes"])
            dependents = st.selectbox("Has Dependents", ["No","Yes"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("<div class='section-label'>Layanan</div>", unsafe_allow_html=True)
            phone_svc   = st.selectbox("Phone Service", ["Yes","No"])
            multi_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
            internet    = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
            online_sec  = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
            online_bak  = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
            dev_prot    = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])

        with c3:
            st.markdown("<div class='section-label'>Billing</div>", unsafe_allow_html=True)
            tech_sup   = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
            stream_tv  = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
            stream_mv  = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
            contract   = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless  = st.selectbox("Paperless Billing", ["Yes","No"])
            payment    = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])
            monthly_ch = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)

        submitted = st.form_submit_button("🔍  Prediksi Churn", use_container_width=True)

    if submitted:
        total_ch = monthly_ch * max(tenure, 1)
        if tenure <= 12:   tg = "0-12"
        elif tenure <= 24: tg = "13-24"
        elif tenure <= 48: tg = "25-48"
        elif tenure <= 60: tg = "49-60"
        else:              tg = "61+"

        raw = {
            "gender":gender, "SeniorCitizen":1 if senior=="Yes" else 0,
            "Partner":partner, "Dependents":dependents, "tenure":tenure,
            "PhoneService":phone_svc, "MultipleLines":multi_lines,
            "InternetService":internet, "OnlineSecurity":online_sec,
            "OnlineBackup":online_bak, "DeviceProtection":dev_prot,
            "TechSupport":tech_sup, "StreamingTV":stream_tv,
            "StreamingMovies":stream_mv, "Contract":contract,
            "PaperlessBilling":paperless, "PaymentMethod":payment,
            "MonthlyCharges":monthly_ch, "TotalCharges":total_ch,
            "tenure_group":tg,
        }
        df_raw = pd.DataFrame([raw])
        df_raw["gender"] = df_raw["gender"].map({"Male":1,"Female":0})
        for col in ["Partner","Dependents","PhoneService","PaperlessBilling"]:
            df_raw[col] = df_raw[col].map({"Yes":1,"No":0})
        df_raw["num_services"] = sum(
            (df_raw[c]=="Yes").astype(int) for c in SERVICE_COLS
        )
        df_enc   = pd.get_dummies(df_raw, columns=ONE_HOT_COLS, drop_first=True)
        df_final = df_enc.reindex(columns=FEATURE_COLS, fill_value=0)

        proba    = model.predict_proba(df_final)[0, 1]
        is_churn = proba >= threshold

        st.markdown("---")
        col1, col2 = st.columns([1, 2])

        with col1:
            color  = RED if is_churn else GREEN
            label  = f"⚠️ BERISIKO CHURN" if is_churn else "✅ AMAN"
            sublbl = f"Probabilitas ≥ threshold {threshold:.2f}" if is_churn else f"Probabilitas < threshold {threshold:.2f}"

            st.markdown(f"""
            <div class="card" style="text-align:center; border-color:{color}40">
                <div style="font-size:42px; font-weight:800; color:{color}">
                    {proba:.1%}
                </div>
                <div style="font-size:14px; font-weight:600; color:{color}; margin-top:4px">
                    {label}
                </div>
                <div style="font-size:11px; color:#8892a4; margin-top:6px">
                    {sublbl}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Gauge bar
            fig_g, ax_g = plt.subplots(figsize=(4, 0.8))
            ax_g.barh([""], [1], color="#2d3748", height=0.5)
            ax_g.barh([""], [proba], color=RED if is_churn else GREEN, height=0.5)
            ax_g.axvline(threshold, color="#e2e8f0", linewidth=1.5,
                         linestyle="--", label=f"threshold={threshold:.2f}")
            ax_g.set_xlim(0, 1)
            ax_g.set_xticks([0, threshold, 1])
            ax_g.set_xticklabels(["0", f"{threshold:.2f}", "1"], fontsize=8)
            ax_g.legend(fontsize=7, loc="lower right", framealpha=0.2,
                        edgecolor="#2d3748")
            fig_g.tight_layout(pad=0.5)
            st.pyplot(fig_g)

        with col2:
            st.markdown("#### Faktor Pendorong Prediksi")
            st.caption("Koefisien × Nilai Fitur — menunjukkan kontribusi spesifik untuk customer ini.")

            coef     = pd.Series(model.coef_[0], index=FEATURE_COLS)
            contrib  = coef * df_final.iloc[0]
            top_cont = contrib.abs().sort_values(ascending=False).head(10).index
            top_c    = contrib[top_cont].sort_values()

            fig_c, ax_c = plt.subplots(figsize=(5.5, 4.5))
            colors_c = [RED if v > 0 else ACCENT for v in top_c.values]
            ax_c.barh(range(len(top_c)), top_c.values,
                      color=colors_c, height=0.6, edgecolor="none")
            ax_c.set_yticks(range(len(top_c)))
            ax_c.set_yticklabels(top_c.index, fontsize=9)
            ax_c.axvline(0, color="#4a5568", linewidth=1)
            ax_c.set_xlabel("Kontribusi terhadap log-odds churn", labelpad=10)
            ax_c.set_title("Merah = dorong ke Churn  |  Biru = dorong ke No Churn",
                           pad=12, fontsize=10)
            ax_c.spines[["top","right"]].set_visible(False)
            ax_c.grid(axis="x", alpha=0.3)
            fig_c.tight_layout()
            st.pyplot(fig_c)