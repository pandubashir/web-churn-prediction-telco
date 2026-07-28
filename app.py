"""
TelcoSight — Customer Churn Intelligence Platform
Run: streamlit run "app churn.py"
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt

# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="TelcoSight · Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

USD_TO_IDR = 17_000

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body { font-family: 'Inter', sans-serif; }
#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header { background: transparent !important; }
[data-testid="stToolbar"] {visibility:hidden;}
[data-testid="stDecoration"] {display:none;}

/* Sidebar tidak dipakai lagi — sembunyikan sepenuhnya */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.stApp { background-color: #080c14; }

[data-testid="stSidebar"] {
    background-color: #0d1220;
    border-right: 1px solid #1a2540;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 18px 20px;
}
[data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 600 !important;
    color: #64748b !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #f1f5f9 !important;
}

h1,h2,h3 { color: #f1f5f9 !important; }

[data-testid="stDataFrame"] {
    border: 1px solid #1a2540;
    border-radius: 10px;
    overflow: hidden;
}

.stAlert { border-radius: 10px; border: none; }

/* Developer badge */
.dev-badge {
    position: fixed;
    top: 14px;
    right: 20px;
    background: linear-gradient(135deg, #1e2d4a, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 11px;
    font-weight: 600;
    color: #60a5fa;
    z-index: 9999;
    letter-spacing: 0.02em;
}

/* Hero */
.hero-brand {
    font-size: 13px;
    font-weight: 700;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 10px;
    text-align: center;
}
.hero-title {
    font-size: 48px;
    font-weight: 800;
    line-height: 1.2;
    margin-bottom: 0;
    text-align: center;
    background: linear-gradient(90deg, #f1f5f9 0%, #60a5fa 50%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 13px;
    color: #475569;
    font-style: italic;
    margin-top: 10px;
    text-align: center;
}

/* Cards */
.card {
    background: linear-gradient(135deg, #111827 0%, #141e30 100%);
    border: 1px solid #1a2540;
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 12px;
}
.card-red    { border-color: #ef444440; }
.card-yellow { border-color: #f59e0b40; }
.card-green  { border-color: #22c55e40; }

.section-label {
    font-size: 10px;
    font-weight: 700;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 10px;
}

/* Probability */
.proba-number { font-size: 56px; font-weight: 800; line-height: 1; }
.proba-high   { color: #ef4444; }
.proba-medium { color: #f59e0b; }
.proba-low    { color: #22c55e; }
.risk-high    { color: #ef4444; font-weight: 800; font-size: 13px; }
.risk-medium  { color: #f59e0b; font-weight: 800; font-size: 13px; }
.risk-low     { color: #22c55e; font-weight: 800; font-size: 13px; }

/* Signal rows */
.signal-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1a2540;
    font-size: 13px;
}
.signal-label        { color: #94a3b8; }
.signal-value-red    { color: #ef4444; font-weight: 600; }
.signal-value-green  { color: #22c55e; font-weight: 600; }
.signal-value-normal { color: #f1f5f9; font-weight: 500; }

/* Rekomendasi */
.rekom-box {
    background: #1a2035;
    border-left: 3px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-size: 13px;
    color: #cbd5e1;
    margin-top: 10px;
}

/* Sidebar detail cards */
.detail-card {
    background: #111827;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}
.detail-label { color: #64748b; font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; }
.detail-value { color: #f1f5f9; font-weight: 600; font-size: 13px; margin-top: 2px; }

.divider { height: 1px; background: #1a2540; margin: 14px 0; }

/* Expander */
[data-testid="stExpander"] {
    background: #111827;
    border: 1px solid #1a2540 !important;
    border-radius: 10px !important;
}

/* Footer row */
.footer-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px 4px 8px 4px;
    border-top: 1px solid #1a2540;
    margin-top: 20px;
    flex-wrap: wrap;
    gap: 8px;
}
.footer-text {
    font-size: 11px;
    color: #1e2d4a;
}

/* Subtle technical-details toggle button */
div[data-testid="stButton"] button[kind="secondary"].tech-toggle-btn,
.tech-toggle-wrap div[data-testid="stButton"] button {
    background: transparent !important;
    border: 1px solid #1a2540 !important;
    color: #3b4a6b !important;
    font-size: 10px !important;
    font-weight: 600 !important;
    padding: 3px 10px !important;
    border-radius: 20px !important;
    letter-spacing: 0.03em;
    box-shadow: none !important;
    min-height: 0 !important;
    line-height: 1.6 !important;
}
.tech-toggle-wrap div[data-testid="stButton"] button:hover {
    border-color: #3b82f6 !important;
    color: #60a5fa !important;
}
</style>
""", unsafe_allow_html=True)

# Developer badge
st.markdown("""
<div class="dev-badge">⚡ Developed by Pandu Bashir Alamin</div>
""", unsafe_allow_html=True)

# Matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor": "#111827", "axes.facecolor": "#111827",
    "axes.edgecolor": "#1e2d4a", "axes.labelcolor": "#94a3b8",
    "axes.titlecolor": "#94a3b8", "xtick.color": "#64748b",
    "ytick.color": "#64748b", "text.color": "#f1f5f9",
    "grid.color": "#1e2d4a", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "font.size": 10,
})

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
model, metadata         = load_model_and_meta()
MODEL_NAME        = metadata["model_name"]
DEFAULT_THRESHOLD = metadata["threshold"]
FEATURE_COLS      = X_test.columns.tolist()
y_proba_all       = model.predict_proba(X_test)[:, 1]

SERVICE_COLS = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]
ONE_HOT_COLS = [
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaymentMethod","tenure_group"
]

# ============================================================
# Helpers
# ============================================================

def fmt_idr(usd_val):
    """Format USD ke Rupiah dengan titik pemisah ribuan (format Indonesia)."""
    idr = int(usd_val * USD_TO_IDR)
    return "Rp {:,}".format(idr).replace(",", ".")

def encode_input(p):
    tenure = p["tenure"]
    if tenure <= 12:   tg = "0-12"
    elif tenure <= 24: tg = "13-24"
    elif tenure <= 48: tg = "25-48"
    elif tenure <= 60: tg = "49-60"
    else:              tg = "61+"

    raw = {
        "gender":        1 if p["gender"] == "Male" else 0,
        "SeniorCitizen": 1 if p["senior"] == "Yes" else 0,
        "Partner":       1 if p["partner"] == "Yes" else 0,
        "Dependents":    1 if p["dependents"] == "Yes" else 0,
        "tenure":        tenure,
        "PhoneService":  1 if p["phone_svc"] == "Yes" else 0,
        "MultipleLines": p["multi_lines"],
        "InternetService": p["internet"],
        "OnlineSecurity":  p["online_sec"],
        "OnlineBackup":    p["online_bak"],
        "DeviceProtection":p["dev_prot"],
        "TechSupport":     p["tech_sup"],
        "StreamingTV":     p["stream_tv"],
        "StreamingMovies": p["stream_mv"],
        "Contract":        p["contract"],
        "PaperlessBilling":1 if p["paperless"] == "Yes" else 0,
        "PaymentMethod":   p["payment"],
        "MonthlyCharges":  p["monthly_ch"],
        "TotalCharges":    p["monthly_ch"] * max(tenure, 1),
        "tenure_group":    tg,
    }
    df = pd.DataFrame([raw])
    df["num_services"] = sum((df[c] == "Yes").astype(int) for c in SERVICE_COLS)

    # Encode manual, bukan pakai get_dummies, karena get_dummies pada
    # dataframe 1 baris akan salah drop kategori (drop_first menganggap
    # nilai yang ada sebagai "kategori pertama" lalu membuangnya, sehingga
    # one-hot jadi 0 semua meski nilainya sebenarnya ada).
    df_final = pd.DataFrame(0, index=df.index, columns=FEATURE_COLS)
    for col in df.columns:
        if col in FEATURE_COLS:
            df_final[col] = df[col]
        elif col in ONE_HOT_COLS:
            dummy_col = f"{col}_{df[col].iloc[0]}"
            if dummy_col in FEATURE_COLS:
                df_final[dummy_col] = 1
    return df_final

def get_risk_level(proba, threshold):
    if proba >= 0.6:
        return "HIGH RISK",   "proba-high",   "card-red",    "risk-high"
    elif proba >= threshold:
        return "MEDIUM RISK", "proba-medium", "card-yellow", "risk-medium"
    else:
        return "LOW RISK",    "proba-low",    "card-green",  "risk-low"

# ============================================================
# Contoh Profil
# ============================================================

SAMPLE_PROFILES = {
    "🔴  HIGH RISK — Pelanggan Baru, Kontrak Bulanan": {
        "gender":"Female","senior":"No","partner":"No","dependents":"No",
        "tenure":3,"phone_svc":"Yes","multi_lines":"No",
        "internet":"Fiber optic","online_sec":"No","online_bak":"No",
        "dev_prot":"No","tech_sup":"No","stream_tv":"No","stream_mv":"No",
        "contract":"Month-to-month","paperless":"Yes",
        "payment":"Electronic check","monthly_ch":89,
        "desc":"Customer baru (3 bulan), Fiber optic tanpa layanan tambahan, "
               "kontrak bulanan, bayar via electronic check. Kombinasi berisiko tinggi.",
        "rekom":"Hubungi dalam 48 jam. Tawarkan diskon upgrade ke kontrak tahunan "
                "atau paket bundling dengan online security gratis 3 bulan."
    },
    "🟡  MEDIUM RISK — Pelanggan Menengah, Mulai Tidak Aktif": {
        "gender":"Male","senior":"No","partner":"Yes","dependents":"No",
        "tenure":18,"phone_svc":"Yes","multi_lines":"Yes",
        "internet":"DSL","online_sec":"No","online_bak":"Yes",
        "dev_prot":"No","tech_sup":"No","stream_tv":"Yes","stream_mv":"No",
        "contract":"Month-to-month","paperless":"Yes",
        "payment":"Mailed check","monthly_ch":60,
        "desc":"Customer 18 bulan dengan DSL, beberapa layanan tambahan tapi "
               "masih kontrak bulanan. Perlu monitoring lebih lanjut.",
        "rekom":"Kirim email personalisasi dengan highlight fitur yang belum dipakai. "
                "Tawarkan loyalty reward untuk upgrade ke kontrak tahunan."
    },
    "🟢  LOW RISK — Pelanggan Loyal, Kontrak Panjang": {
        "gender":"Male","senior":"No","partner":"Yes","dependents":"Yes",
        "tenure":62,"phone_svc":"Yes","multi_lines":"Yes",
        "internet":"DSL","online_sec":"Yes","online_bak":"Yes",
        "dev_prot":"Yes","tech_sup":"Yes","stream_tv":"Yes","stream_mv":"Yes",
        "contract":"Two year","paperless":"No",
        "payment":"Credit card (automatic)","monthly_ch":79,
        "desc":"Customer loyal 62 bulan dengan kontrak 2 tahun, semua layanan "
               "tambahan aktif, bayar otomatis via credit card. Sangat stabil.",
        "rekom":"Tidak perlu intervensi khusus. Kirim appreciation email tahunan "
                "dan pastikan renewal kontrak berjalan mulus."
    },
}

# ============================================================
# Global metrics
# ============================================================

total_customers = len(y_test)
churn_rate      = y_test.mean()
at_risk         = (y_proba_all >= DEFAULT_THRESHOLD).sum()
avg_mrr_risk    = X_test.loc[y_proba_all >= DEFAULT_THRESHOLD, "MonthlyCharges"].mean()
mrr_at_risk     = at_risk * avg_mrr_risk

# ============================================================
# Session State
# ============================================================

if "show_model_detail" not in st.session_state:
    st.session_state["show_model_detail"] = False

# ============================================================
# HERO
# ============================================================

st.markdown("""
<div style="padding:16px 0 24px 0">
    <div class="hero-brand">📊 TelcoSight · Business Intelligence</div>
    <div class="hero-title">Customer Churn Intelligence Platform</div>
    <div class="hero-sub">
        Identifikasi pelanggan berisiko sebelum mereka pergi.
        Prediksi real-time berbasis data pelanggan Telco.
    </div>
</div>
""", unsafe_allow_html=True)

# Global metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pelanggan",  f"{total_customers:,}",
          help="Jumlah customer di test set")
c2.metric("Churn Rate Aktual", f"{churn_rate:.1%}",
          help="Proporsi customer yang benar-benar churn di test set")
c3.metric("Pelanggan At Risk", f"{at_risk:,}",
          help=f"Customer diprediksi churn (threshold ≥ {DEFAULT_THRESHOLD})")
c4.metric("MRR At Risk",       fmt_idr(mrr_at_risk),
          help="Estimasi monthly revenue yang terancam dari customer at risk")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ============================================================
# INPUT  &  HASIL — side by side
# ============================================================

col_input, col_hasil = st.columns([1, 1], gap="large")

# ---------- KOLOM INPUT ----------
with col_input:
    st.markdown("<div class='section-label'>Input Data Pelanggan</div>",
                unsafe_allow_html=True)

    input_mode = st.radio(
        "Mode", ["✏️ Input Manual", "📋 Contoh Profil"],
        horizontal=True, label_visibility="collapsed"
    )

    current_profile = {}

    # ---- CONTOH PROFIL ----
    if input_mode == "📋 Contoh Profil":
        selected = st.selectbox(
            "Pilih profil contoh:", list(SAMPLE_PROFILES.keys()),
            label_visibility="collapsed"
        )
        prof = SAMPLE_PROFILES[selected]

        card_cls = ("card-red"    if "HIGH"   in selected else
                    "card-yellow" if "MEDIUM" in selected else "card-green")
        sec_color = "#ef4444" if prof["online_sec"] == "No" else "#22c55e"

        st.markdown(f"""
        <div class="card {card_cls}">
            <div class="section-label">Profil Customer</div>
            <div style="font-size:12px; color:#94a3b8; margin-bottom:12px">
                {prof['desc']}
            </div>
            <div class="signal-row">
                <span class="signal-label">Tenure</span>
                <span class="signal-value-normal">{prof['tenure']} bulan</span>
            </div>
            <div class="signal-row">
                <span class="signal-label">Contract</span>
                <span class="signal-value-normal">{prof['contract']}</span>
            </div>
            <div class="signal-row">
                <span class="signal-label">Internet Service</span>
                <span class="signal-value-normal">{prof['internet']}</span>
            </div>
            <div class="signal-row">
                <span class="signal-label">Payment Method</span>
                <span class="signal-value-normal">{prof['payment']}</span>
            </div>
            <div class="signal-row">
                <span class="signal-label">Monthly Charges</span>
                <span class="signal-value-normal">
                    ${prof['monthly_ch']:,} ({fmt_idr(prof['monthly_ch'])})
                </span>
            </div>
            <div class="signal-row" style="border:none">
                <span class="signal-label">Online Security</span>
                <span style="color:{sec_color}; font-weight:600">
                    {prof['online_sec']}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        current_profile = prof

    # ---- INPUT MANUAL ----
    else:
        OPTS = {
            "MultipleLines":    ["No","Yes","No phone service"],
            "InternetService":  ["DSL","Fiber optic","No"],
            "OnlineSecurity":   ["No","Yes","No internet service"],
            "OnlineBackup":     ["No","Yes","No internet service"],
            "DeviceProtection": ["No","Yes","No internet service"],
            "TechSupport":      ["No","Yes","No internet service"],
            "StreamingTV":      ["No","Yes","No internet service"],
            "StreamingMovies":  ["No","Yes","No internet service"],
            "Contract":         ["Month-to-month","One year","Two year"],
            "PaymentMethod":    [
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ],
        }

        with st.expander("👤 Demografi", expanded=True):
            g1, g2 = st.columns(2)
            gender = g1.selectbox(
                "Gender", ["Female","Male"],
                help="Jenis kelamin pelanggan"
            )
            senior = g2.selectbox(
                "Senior Citizen", ["No","Yes"],
                help="Apakah pelanggan berusia 65 tahun ke atas?"
            )
            g3, g4 = st.columns(2)
            partner = g3.selectbox(
                "Partner", ["No","Yes"],
                help="Apakah pelanggan memiliki pasangan?"
            )
            dependents = g4.selectbox(
                "Dependents", ["No","Yes"],
                help="Apakah pelanggan memiliki tanggungan (anak, orang tua, dll)?"
            )
            tenure = st.slider(
                "Tenure (bulan)", 0, 72, 12,
                help="Lama pelanggan telah berlangganan dalam bulan. "
                     "Semakin lama biasanya semakin loyal."
            )

        with st.expander("📡 Layanan", expanded=True):
            s1, s2 = st.columns(2)
            phone_svc = s1.selectbox(
                "Phone Service", ["Yes","No"],
                help="Apakah pelanggan berlangganan layanan telepon?"
            )
            multi_lines = s2.selectbox(
                "Multiple Lines", OPTS["MultipleLines"],
                help="Apakah pelanggan punya lebih dari satu saluran telepon?"
            )
            s3, s4 = st.columns(2)
            internet = s3.selectbox(
                "Internet Service", OPTS["InternetService"],
                help="Jenis layanan internet: DSL (lebih stabil/murah), "
                     "Fiber optic (lebih cepat/mahal), atau tidak berlangganan."
            )
            online_sec = s4.selectbox(
                "Online Security", OPTS["OnlineSecurity"],
                help="Layanan keamanan online (proteksi dari ancaman siber). "
                     "'No internet service' berarti tidak berlangganan internet."
            )
            s5, s6 = st.columns(2)
            online_bak = s5.selectbox(
                "Online Backup", OPTS["OnlineBackup"],
                help="Layanan backup data online untuk melindungi file penting."
            )
            dev_prot = s6.selectbox(
                "Device Protection", OPTS["DeviceProtection"],
                help="Layanan proteksi perangkat (asuransi jika perangkat rusak/hilang)."
            )
            s7, s8 = st.columns(2)
            tech_sup = s7.selectbox(
                "Tech Support", OPTS["TechSupport"],
                help="Layanan dukungan teknis premium dari tim CS khusus."
            )
            stream_tv = s8.selectbox(
                "Streaming TV", OPTS["StreamingTV"],
                help="Apakah pelanggan berlangganan layanan streaming TV?"
            )
            stream_mv = st.selectbox(
                "Streaming Movies", OPTS["StreamingMovies"],
                help="Apakah pelanggan berlangganan layanan streaming film/movie?"
            )

        with st.expander("💳 Billing", expanded=True):
            contract = st.selectbox(
                "Contract", OPTS["Contract"],
                help="Tipe kontrak: Month-to-month (paling fleksibel, risiko churn tinggi), "
                     "One year, atau Two year (paling loyal)."
            )
            b1, b2 = st.columns(2)
            paperless = b1.selectbox(
                "Paperless Billing", ["Yes","No"],
                help="Apakah pelanggan memilih tagihan digital (email) "
                     "dibanding tagihan kertas?"
            )
            payment = b2.selectbox(
                "Payment Method", OPTS["PaymentMethod"],
                help="Metode pembayaran: Electronic check (risiko churn lebih tinggi), "
                     "Mailed check, Bank transfer (automatic), atau Credit card (automatic)."
            )
            monthly_ch = st.number_input(
                "Monthly Charges ($)", 0, 200, 70, step=1,
                help="Total tagihan bulanan pelanggan dalam USD. "
                     "Charges lebih tinggi berkorelasi dengan risiko churn lebih tinggi."
            )
            st.caption(f"≈ {fmt_idr(monthly_ch)} / bulan")

        current_profile = {
            "gender":gender, "senior":senior, "partner":partner,
            "dependents":dependents, "tenure":tenure, "phone_svc":phone_svc,
            "multi_lines":multi_lines, "internet":internet, "online_sec":online_sec,
            "online_bak":online_bak, "dev_prot":dev_prot, "tech_sup":tech_sup,
            "stream_tv":stream_tv, "stream_mv":stream_mv, "contract":contract,
            "paperless":paperless, "payment":payment, "monthly_ch":monthly_ch,
        }

    # Tombol prediksi — selalu harus klik, tidak ada auto-predict
    predict_btn = st.button(
        "🔍  Analisis Risiko Churn",
        use_container_width=True,
        type="primary"
    )

# ---------- KOLOM HASIL ----------
with col_hasil:
    st.markdown("<div class='section-label'>Hasil Analisis Risiko</div>",
                unsafe_allow_html=True)

    if predict_btn and current_profile:
        df_input = encode_input(current_profile)
        proba    = model.predict_proba(df_input)[0, 1]
        risk_lbl, proba_cls, card_cls, risk_cls = get_risk_level(proba, DEFAULT_THRESHOLD)
        monthly_usd = current_profile.get("monthly_ch", 70)
        rekom = current_profile.get(
            "rekom",
            "Lakukan analisis lebih lanjut berdasarkan riwayat interaksi customer."
        )

        # Probabilitas
        st.markdown(f"""
        <div class="card {card_cls}">
            <div class="section-label">Probabilitas Churn</div>
            <div class="{proba_cls} proba-number">{proba:.1%}</div>
            <div class="{risk_cls}" style="margin-top:6px">⚠ {risk_lbl}</div>
            <div style="font-size:11px; color:#475569; margin-top:4px">
                threshold = {DEFAULT_THRESHOLD} ·
                {'DIPREDIKSI CHURN' if proba >= DEFAULT_THRESHOLD else 'AMAN'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Rekomendasi
        st.markdown(f"""
        <div class="card">
            <div class="section-label">🎯 Rekomendasi Tim CS</div>
            <div class="rekom-box">{rekom}</div>
        </div>
        """, unsafe_allow_html=True)

        # Business impact
        bi1, bi2 = st.columns(2)
        bi1.metric("Revenue Berisiko / Bulan", fmt_idr(monthly_usd))
        bi2.metric("Revenue Berisiko / Tahun",  fmt_idr(monthly_usd * 12))

        # Faktor pendorong
        st.markdown(
            "<div class='section-label' style='margin-top:14px'>Detail Sinyal Risiko</div>",
            unsafe_allow_html=True
        )
        # model adalah XGBoost (tree-based) sehingga tidak punya .coef_
        # seperti model linear. Gunakan kontribusi SHAP bawaan XGBoost
        # (pred_contribs) untuk mendapatkan kontribusi tiap fitur pada
        # prediksi ini.
        import xgboost as xgb
        booster = model.get_booster()
        dmat    = xgb.DMatrix(df_input[FEATURE_COLS])
        shap_vals = booster.predict(dmat, pred_contribs=True)[0]
        # kolom terakhir adalah base value (bias), buang
        contrib = pd.Series(shap_vals[:-1], index=FEATURE_COLS)
        top_idx = contrib.abs().sort_values(ascending=False).head(8).index
        top_c   = contrib[top_idx].sort_values()

        fig, ax = plt.subplots(figsize=(5, 3.5))
        colors  = ["#ef4444" if v > 0 else "#3b82f6" for v in top_c.values]
        ax.barh(range(len(top_c)), top_c.values,
                color=colors, height=0.55, edgecolor="none")
        ax.set_yticks(range(len(top_c)))
        ax.set_yticklabels(top_c.index, fontsize=8.5)
        ax.axvline(0, color="#374151", linewidth=1)
        ax.set_xlabel("Kontribusi ke log-odds churn", fontsize=9, labelpad=8)
        ax.set_title("🔴 Dorong Churn  |  🔵 Dorong No Churn",
                     fontsize=9, pad=10, color="#94a3b8")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout(pad=0.8)
        st.pyplot(fig)

    else:
        st.markdown("""
        <div class="card" style="text-align:center; padding:48px 20px">
            <div style="font-size:38px">🔍</div>
            <div style="font-size:15px; font-weight:600;
                        color:#f1f5f9; margin-top:14px">
                Belum ada prediksi
            </div>
            <div style="font-size:12px; color:#64748b; margin-top:6px">
                Pilih contoh profil atau isi input manual,<br>
                lalu klik <b>Analisis Risiko Churn</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# SEGMENTASI BAWAH
# ============================================================

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-label'>Segmentasi Risiko — Seluruh Customer</div>",
    unsafe_allow_html=True
)

risk_df = pd.DataFrame({
    "actual_churn":    y_test.values,
    "churn_proba":     y_proba_all,
    "monthly_charges": X_test["MonthlyCharges"].values
})

def seg_all(p):
    if p >= 0.6:                 return "🔴 High Risk"
    elif p >= DEFAULT_THRESHOLD: return "🟡 Medium Risk"
    else:                        return "🟢 Low Risk"

risk_df["segment"] = risk_df["churn_proba"].apply(seg_all)
summary = risk_df.groupby("segment").agg(
    Jumlah      = ("actual_churn","count"),
    Churn_Rate  = ("actual_churn","mean"),
    Avg_Monthly = ("monthly_charges","mean")
).reindex(["🔴 High Risk","🟡 Medium Risk","🟢 Low Risk"]).dropna()

r1, r2, r3 = st.columns(3)
for col_st, (seg, row) in zip([r1,r2,r3], summary.iterrows()):
    color   = "#ef4444" if "High" in seg else "#f59e0b" if "Medium" in seg else "#22c55e"
    mrr_est = fmt_idr(row["Jumlah"] * row["Avg_Monthly"] * row["Churn_Rate"])
    col_st.markdown(f"""
    <div class="card" style="border-color:{color}40">
        <div style="font-size:13px; font-weight:700; color:{color}">{seg}</div>
        <div style="font-size:30px; font-weight:800; color:#f1f5f9; margin:8px 0">
            {int(row['Jumlah']):,}
        </div>
        <div style="font-size:11px; color:#64748b; margin-bottom:10px">customer</div>
        <div class="signal-row">
            <span class="signal-label">Actual Churn Rate</span>
            <span style="color:{color}; font-weight:700">{row['Churn_Rate']:.1%}</span>
        </div>
        <div class="signal-row">
            <span class="signal-label">Avg Monthly</span>
            <span class="signal-value-normal">{fmt_idr(row['Avg_Monthly'])}</span>
        </div>
        <div class="signal-row" style="border:none">
            <span class="signal-label">MRR At Risk</span>
            <span style="color:{color}; font-weight:600">{mrr_est}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER — dengan tombol "Technical Details" tersembunyi/subtle
# ============================================================

st.markdown("<div style='border-top:1px solid #1a2540; margin-top:20px'></div>",
            unsafe_allow_html=True)

f1, f2 = st.columns([5, 1.3])
with f1:
    st.markdown("""
    <div style="padding-top:14px; font-size:11px; color:#1e2d4a">
        TelcoSight · Dataset: IBM Telco Customer Churn ·
        Model: XGBoost · Stack: Python · Scikit-learn · Streamlit ·
        ⚡ Developed by Pandu Bashir Alamin
    </div>
    """, unsafe_allow_html=True)
with f2:
    st.markdown('<div class="tech-toggle-wrap" style="padding-top:10px; text-align:right">',
                unsafe_allow_html=True)
    if st.button("🔬 Technical Details", key="tech_toggle"):
        st.session_state["show_model_detail"] = not st.session_state["show_model_detail"]
    st.markdown('</div>', unsafe_allow_html=True)

# Section detail teknis — hanya muncul kalau tombol di footer diklik.
# Ditujukan untuk data scientist / interviewer / technical reviewer,
# tidak mengganggu tampilan default untuk user bisnis (CS/retention).
if st.session_state["show_model_detail"]:
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-label'>🔬 Technical Details — Model Diagnostics</div>",
        unsafe_allow_html=True
    )

    with st.expander("Detail Model Teknis", expanded=True):
        d1, d2, d3 = st.columns(3)
        details = [
            ("Model",             MODEL_NAME),
            ("ROC-AUC",           str(metadata["roc_auc"])),
            ("Recall (optimal)",  f"{metadata['recall']:.1%}"),
            ("Threshold",         str(DEFAULT_THRESHOLD)),
            ("False Negative",    f"{metadata['fn']} customer terlewat"),
            ("False Positive",    f"{metadata['fp']} false alarm"),
        ]
        for i, (label, value) in enumerate(details):
            target = [d1, d2, d3][i % 3]
            target.markdown(f"""
            <div class='detail-card'>
                <div class='detail-label'>{label}</div>
                <div class='detail-value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        tt1, tt2 = st.columns([1.1, 1])

        with tt1:
            # Threshold trade-off table
            st.markdown("<div class='section-label'>Threshold Trade-off</div>",
                        unsafe_allow_html=True)
            from sklearn.metrics import confusion_matrix as cm_fn, roc_curve, auc as auc_fn

            rows_thr = []
            for t in np.arange(0.20, 0.55, 0.05):
                pt  = (y_proba_all >= t).astype(int)
                cmt = cm_fn(y_test, pt)
                tnt, fpt, fnt, tpt = cmt.ravel()
                rec  = tpt / (tpt + fnt) if (tpt + fnt) > 0 else 0
                prec = tpt / (tpt + fpt) if (tpt + fpt) > 0 else 0
                rows_thr.append({
                    "T":         round(t, 2),
                    "Recall":    f"{rec:.0%}",
                    "Precision": f"{prec:.0%}",
                    "FN":        fnt,
                    "":          "✅" if abs(t - DEFAULT_THRESHOLD) < 0.01 else ""
                })
            st.dataframe(pd.DataFrame(rows_thr), use_container_width=True, hide_index=True)

            # Confusion matrix pada threshold aktif
            st.markdown("<div class='section-label' style='margin-top:10px'>"
                        "Confusion Matrix (threshold aktif)</div>",
                        unsafe_allow_html=True)
            pred_active = (y_proba_all >= DEFAULT_THRESHOLD).astype(int)
            cm_active = cm_fn(y_test, pred_active)
            cm_df = pd.DataFrame(
                cm_active,
                index=["Actual: No Churn", "Actual: Churn"],
                columns=["Pred: No Churn", "Pred: Churn"]
            )
            st.dataframe(cm_df, use_container_width=True)

        with tt2:
            # ROC Curve
            st.markdown("<div class='section-label'>ROC Curve</div>", unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_test, y_proba_all)
            roc_auc_val = auc_fn(fpr, tpr)
            fig_roc, ax_roc = plt.subplots(figsize=(3.6, 2.6))
            ax_roc.plot(fpr, tpr, color="#3b82f6", linewidth=2,
                        label=f"AUC={roc_auc_val:.3f}")
            ax_roc.fill_between(fpr, tpr, alpha=0.07, color="#3b82f6")
            ax_roc.plot([0,1],[0,1], "--", color="#374151", linewidth=1)
            ax_roc.set_xlabel("FPR", fontsize=8)
            ax_roc.set_ylabel("TPR", fontsize=8)
            ax_roc.legend(fontsize=8, framealpha=0.15)
            ax_roc.spines[["top","right"]].set_visible(False)
            fig_roc.tight_layout(pad=0.5)
            st.pyplot(fig_roc)

        st.caption(
            "Section ini ditujukan untuk technical reviewer / data scientist. "
            "Klik kembali tombol 🔬 Technical Details di footer untuk menyembunyikan."
        )