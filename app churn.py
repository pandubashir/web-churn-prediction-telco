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
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

sns.set_style("whitegrid")

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# Load metadata, data, model
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
model, metadata = load_model_and_meta()
MODEL_NAME    = metadata["model_name"]
DEFAULT_THRESHOLD = metadata["threshold"]
FEATURE_COLS  = X_test.columns.tolist()

y_proba_default = model.predict_proba(X_test)[:, 1]

# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("📊 Churn Prediction")
st.sidebar.markdown(
    f"**Model:** {MODEL_NAME}  \n"
    f"**ROC-AUC:** {metadata['roc_auc']}  \n"
    f"**Recall (optimal):** {metadata['recall']:.1%}  \n"
    f"**Optimal threshold:** {DEFAULT_THRESHOLD}"
)
st.sidebar.markdown("---")
threshold = st.sidebar.slider(
    "Classification Threshold",
    min_value=0.10, max_value=0.70,
    value=float(DEFAULT_THRESHOLD), step=0.05,
    help=(
        "Probabilitas minimum untuk mengklasifikasikan customer sebagai Churn. "
        "Threshold lebih rendah = recall lebih tinggi, precision lebih rendah."
    )
)
st.sidebar.caption(
    f"Threshold optimal dari training: **{DEFAULT_THRESHOLD}** "
    f"(dipilih berdasarkan F2-score tertinggi, recall={metadata['recall']:.1%})"
)

y_pred = (y_proba_default >= threshold).astype(int)

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Model Performance",
    "💰 Risk Segmentation & Business Impact",
    "🔍 Model Interpretation",
    "🧑 Predict New Customer"
])

# ============================================================
# TAB 1 — Model Performance
# ============================================================

with tab1:
    st.header(f"Model Performance — {MODEL_NAME}")

    cm   = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy  = (tp + tn) / len(y_test)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall (Churn)", f"{recall:.1%}",
               help="Dari semua yang benar-benar churn, berapa persen terdeteksi.")
    c2.metric("Precision (Churn)", f"{precision:.1%}",
               help="Dari semua yang diprediksi churn, berapa persen benar.")
    c3.metric("F1-Score", f"{f1:.3f}")
    c4.metric("ROC-AUC", f"{metadata['roc_auc']}")

    st.info(
        f"Pada threshold **{threshold:.2f}**: model mendeteksi **{tp}** dari "
        f"**{tp+fn}** customer churn (recall {recall:.1%}), "
        f"dengan **{fp}** false alarm dan **{fn}** customer churn yang terlewat."
    )

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred: No Churn", "Pred: Churn"],
                    yticklabels=["Actual: No Churn", "Actual: Churn"])
        ax.set_title(f"threshold = {threshold:.2f}")
        st.pyplot(fig)

    with col_r:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_proba_default)
        roc_auc     = auc(fpr, tpr)
        fig2, ax2   = plt.subplots(figsize=(4.5, 3.5))
        ax2.plot(fpr, tpr, label=f"{MODEL_NAME} (AUC={roc_auc:.3f})")
        ax2.plot([0,1],[0,1], "--", color="gray", label="Random Guess")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.legend()
        st.pyplot(fig2)

    st.subheader("Trade-off di Berbagai Threshold")
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
            "Dipilih": "✅" if t == DEFAULT_THRESHOLD else ""
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ============================================================
# TAB 2 — Risk Segmentation & Business Impact
# ============================================================

with tab2:
    st.header("Risk Segmentation & Business Impact")

    risk_df = pd.DataFrame({
        "actual_churn":    y_test.values,
        "churn_proba":     y_proba_default,
        "monthly_charges": X_test["MonthlyCharges"].values
    })

    c1, c2 = st.columns(2)
    with c1:
        high_cut = st.slider("High Risk (≥)", 0.5, 0.9, 0.6, 0.05)
    with c2:
        med_cut  = st.slider("Medium Risk (≥)", 0.2, float(high_cut)-0.05,
                              DEFAULT_THRESHOLD, 0.05)

    def segment(p):
        if p >= high_cut:   return "High Risk"
        elif p >= med_cut:  return "Medium Risk"
        else:               return "Low Risk"

    risk_df["segment"] = risk_df["churn_proba"].apply(segment)

    summary = risk_df.groupby("segment").agg(
        Jumlah_Customer   = ("actual_churn", "count"),
        Actual_Churn_Rate = ("actual_churn", "mean"),
        Avg_Monthly_Charges = ("monthly_charges", "mean")
    ).reindex(["High Risk", "Medium Risk", "Low Risk"]).dropna()

    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.subheader("Segment Summary")
        st.dataframe(
            summary.style.format({
                "Actual_Churn_Rate": "{:.1%}",
                "Avg_Monthly_Charges": "${:.2f}"
            }),
            use_container_width=True
        )

        fig, ax = plt.subplots(figsize=(4.5, 3))
        colors = {"High Risk": "#d62728", "Medium Risk": "#ff7f0e", "Low Risk": "#2ca02c"}
        bars = ax.bar(
            summary.index,
            summary["Actual_Churn_Rate"],
            color=[colors.get(s, "gray") for s in summary.index]
        )
        ax.set_ylabel("Actual Churn Rate")
        ax.set_ylim(0, 1)
        for bar, val in zip(bars, summary["Actual_Churn_Rate"]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                    f"{val:.1%}", ha="center", fontweight="bold")
        st.pyplot(fig)

    with col_r:
        st.subheader("💰 Business Impact Calculator")
        success_rate = st.slider(
            "Retention campaign success rate", 0.0, 1.0, 0.30, 0.05,
            help="% customer High Risk yang benar-benar akan churn namun berhasil diretensi."
        )

        high_risk       = risk_df[risk_df["segment"] == "High Risk"]
        high_risk_churn = high_risk[high_risk["actual_churn"] == 1]

        n_saved      = len(high_risk_churn) * success_rate
        monthly_rev  = n_saved * high_risk_churn["monthly_charges"].mean() if len(high_risk_churn) > 0 else 0
        annual_rev   = monthly_rev * 12

        st.metric("Customer High Risk", f"{len(high_risk)}")
        st.metric("...yang benar-benar churn", f"{len(high_risk_churn)}")
        st.metric("Estimasi customer terselamatkan", f"{n_saved:.1f}")
        st.metric("Revenue terselamatkan / tahun", f"${annual_rev:,.2f}")
        st.caption("Angka ilustratif — success rate aktual tergantung efektivitas campaign.")

# ============================================================
# TAB 3 — Model Interpretation
# ============================================================

with tab3:
    st.header(f"Model Interpretation — {MODEL_NAME}")
    st.markdown(
        "Untuk **Logistic Regression**, interpretasi terbaik adalah melalui "
        "**koefisien model** dan **odds ratio** — bukan feature importance "
        "(khusus tree-based) atau SHAP TreeExplainer (khusus tree-based)."
    )

    coef = pd.Series(model.coef_[0], index=FEATURE_COLS)
    top_idx  = coef.abs().sort_values(ascending=False).head(15).index
    top_coef = coef[top_idx].sort_values()
    odds_ratio = np.exp(coef[top_idx]).sort_values(ascending=False)

    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("Koefisien (Log-Odds)")
        st.caption("Merah = meningkatkan risiko churn | Biru = menurunkan risiko churn")
        fig, ax = plt.subplots(figsize=(5, 5))
        colors  = ["#d62728" if v > 0 else "#1f77b4" for v in top_coef.values]
        ax.barh(top_coef.index, top_coef.values, color=colors)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Log-Odds Coefficient")
        st.pyplot(fig)

    with col_r:
        st.subheader("Odds Ratio")
        st.caption("OR > 1 = meningkatkan risiko churn | OR < 1 = menurunkan risiko churn")
        or_sorted = odds_ratio.sort_values()
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        colors2   = ["#d62728" if v > 1 else "#1f77b4" for v in or_sorted.values]
        ax2.barh(or_sorted.index, or_sorted.values, color=colors2)
        ax2.axvline(1, color="gray", linewidth=0.8, linestyle="--")
        ax2.set_xlabel("Odds Ratio exp(coef)")
        st.pyplot(fig2)

    st.subheader("Tabel Odds Ratio")
    or_table = pd.DataFrame({
        "Feature":     odds_ratio.index,
        "Coefficient": coef[odds_ratio.index].round(4).values,
        "Odds_Ratio":  odds_ratio.round(3).values,
        "Arah":        ["⬆ Meningkatkan churn" if v > 1 else "⬇ Menurunkan churn"
                        for v in odds_ratio.values]
    })
    st.dataframe(or_table, use_container_width=True, hide_index=True)

    with st.expander("📖 Cara membaca Odds Ratio"):
        st.markdown("""
- **OR > 1**: fitur ini meningkatkan risiko churn.
  Contoh OR = 2.5 → customer dengan fitur ini **2.5x lebih mungkin churn** dibanding baseline.
- **OR < 1**: fitur ini menurunkan risiko churn.
  Contoh OR = 0.3 → customer dengan fitur ini hanya **0.3x** kemungkinan churn (lebih aman).
- **OR = 1**: fitur tidak berpengaruh pada probabilitas churn.
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
    "PaymentMethod":    [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ],
}
ONE_HOT_COLS = list(CATEGORICAL_OPTIONS.keys()) + ["tenure_group"]
SERVICE_COLS = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]

with tab4:
    st.header("Predict New Customer")
    st.markdown(
        "Masukkan profil customer untuk mendapatkan **probabilitas churn** "
        "dan **penjelasan faktor pendorongnya**."
    )

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demografi**")
            gender      = st.selectbox("Gender", ["Female","Male"])
            senior      = st.selectbox("Senior Citizen", ["No","Yes"])
            partner     = st.selectbox("Has Partner", ["No","Yes"])
            dependents  = st.selectbox("Has Dependents", ["No","Yes"])
            tenure      = st.slider("Tenure (months)", 0, 72, 12)

        with c2:
            st.markdown("**Layanan**")
            phone_svc   = st.selectbox("Phone Service", ["Yes","No"])
            multi_lines = st.selectbox("Multiple Lines", CATEGORICAL_OPTIONS["MultipleLines"])
            internet    = st.selectbox("Internet Service", CATEGORICAL_OPTIONS["InternetService"])
            online_sec  = st.selectbox("Online Security", CATEGORICAL_OPTIONS["OnlineSecurity"])
            online_bak  = st.selectbox("Online Backup", CATEGORICAL_OPTIONS["OnlineBackup"])
            dev_prot    = st.selectbox("Device Protection", CATEGORICAL_OPTIONS["DeviceProtection"])

        with c3:
            st.markdown("**Billing**")
            tech_sup    = st.selectbox("Tech Support", CATEGORICAL_OPTIONS["TechSupport"])
            stream_tv   = st.selectbox("Streaming TV", CATEGORICAL_OPTIONS["StreamingTV"])
            stream_mv   = st.selectbox("Streaming Movies", CATEGORICAL_OPTIONS["StreamingMovies"])
            contract    = st.selectbox("Contract", CATEGORICAL_OPTIONS["Contract"])
            paperless   = st.selectbox("Paperless Billing", ["Yes","No"])
            payment     = st.selectbox("Payment Method", CATEGORICAL_OPTIONS["PaymentMethod"])
            monthly_ch  = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, step=1.0)

        submitted = st.form_submit_button("🔍 Prediksi Churn")

    if submitted:
        total_ch = monthly_ch * max(tenure, 1)

        if tenure <= 12:   tg = "0-12"
        elif tenure <= 24: tg = "13-24"
        elif tenure <= 48: tg = "25-48"
        elif tenure <= 60: tg = "49-60"
        else:              tg = "61+"

        raw = {
            "gender": gender, "SeniorCitizen": 1 if senior=="Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone_svc, "MultipleLines": multi_lines,
            "InternetService": internet, "OnlineSecurity": online_sec,
            "OnlineBackup": online_bak, "DeviceProtection": dev_prot,
            "TechSupport": tech_sup, "StreamingTV": stream_tv,
            "StreamingMovies": stream_mv, "Contract": contract,
            "PaperlessBilling": paperless, "PaymentMethod": payment,
            "MonthlyCharges": monthly_ch, "TotalCharges": total_ch,
            "tenure_group": tg,
        }
        df_raw = pd.DataFrame([raw])

        # Encoding (sama persis dengan 02_data_preparation.py)
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
            st.metric("Churn Probability", f"{proba:.1%}")
            if is_churn:
                st.error(f"⚠️ BERISIKO CHURN (≥ {threshold:.2f})")
            else:
                st.success(f"✅ AMAN (< {threshold:.2f})")

            # Gauge bar
            fig_g, ax_g = plt.subplots(figsize=(4, 0.6))
            ax_g.barh([""], [1], color="#e0e0e0", height=0.4)
            ax_g.barh([""], [proba],
                       color="#d62728" if is_churn else "#2ca02c", height=0.4)
            ax_g.axvline(threshold, color="black", linewidth=1.5,
                          linestyle="--", label=f"threshold={threshold:.2f}")
            ax_g.set_xlim(0, 1)
            ax_g.set_xticks([0, threshold, 1])
            ax_g.legend(fontsize=7, loc="lower right")
            ax_g.set_title(f"Probabilitas Churn: {proba:.1%}", fontsize=9)
            st.pyplot(fig_g)

        with col2:
            st.markdown("**Faktor Pendorong Prediksi (Koefisien × Nilai Fitur)**")
            coef     = pd.Series(model.coef_[0], index=FEATURE_COLS)
            contrib  = coef * df_final.iloc[0]
            top_cont = contrib.abs().sort_values(ascending=False).head(10).index
            top_c    = contrib[top_cont].sort_values()

            fig_c, ax_c = plt.subplots(figsize=(5, 4))
            colors_c = ["#d62728" if v > 0 else "#1f77b4" for v in top_c.values]
            ax_c.barh(top_c.index, top_c.values, color=colors_c)
            ax_c.axvline(0, color="gray", linewidth=0.8)
            ax_c.set_xlabel("Kontribusi terhadap log-odds churn")
            ax_c.set_title("Merah = dorong ke Churn | Biru = dorong ke No Churn")
            st.pyplot(fig_c)
            st.caption(
                "Visualisasi ini menunjukkan fitur mana yang paling berkontribusi "
                "pada prediksi untuk customer ini secara spesifik."
            )
