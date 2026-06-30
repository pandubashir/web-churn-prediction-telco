# %% [markdown]
# # 04 - Evaluation & Business Translation
# Perubahan dari versi sebelumnya:
# - Load model final & threshold dari model_metadata.json (tidak hardcode)
# - SHAP diganti dengan koefisien LR (lebih tepat untuk Logistic Regression)
# - Business translation menggunakan threshold optimal dari metadata

# %% Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    classification_report, fbeta_score, precision_recall_curve
)

sns.set_style("whitegrid")

# %% Load metadata model final
with open("data/model_metadata.json", "r") as f:
    metadata = json.load(f)

MODEL_NAME = metadata["model_name"]
THRESHOLD  = metadata["threshold"]

print(f"Model final  : {MODEL_NAME}")
print(f"Threshold    : {THRESHOLD}")
print(f"Recall       : {metadata['recall']:.1%}")
print(f"ROC-AUC      : {metadata['roc_auc']}")
print(f"False Negative: {metadata['fn']}")

# %% Load data & model
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

model   = joblib.load("data/model_final.pkl")
y_proba = np.load("data/y_proba_final.npy")
y_pred  = (y_proba >= THRESHOLD).astype(int)

# ============================================================
# %% [markdown]
# ## 1. Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

plt.figure(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["Pred: No Churn", "Pred: Churn"],
    yticklabels=["Actual: No Churn", "Actual: Churn"]
)
plt.title(f"Confusion Matrix — {MODEL_NAME} (threshold={THRESHOLD})")
plt.tight_layout()
plt.show()

print(f"\nTrue Negative  (benar tidak churn) : {tn}")
print(f"False Positive (false alarm)        : {fp}")
print(f"False Negative (churn terlewat)     : {fn}  ← ingin diminimalkan")
print(f"True Positive  (benar churn)        : {tp}")
print(f"\n{classification_report(y_test, y_pred)}")

# ============================================================
# %% [markdown]
# ## 2. ROC Curve & Precision-Recall Curve
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc_score   = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, label=f"{MODEL_NAME} (AUC={auc_score:.3f})")
axes[0].plot([0,1],[0,1], "--", color="gray", label="Random Guess")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curve")
axes[0].legend()

# Precision-Recall Curve
prec_arr, rec_arr, thr_arr = precision_recall_curve(y_test, y_proba)
axes[1].plot(rec_arr, prec_arr)
axes[1].axvline(metadata["recall"], color="red", linestyle="--",
                label=f"Operating point (recall={metadata['recall']:.1%})")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")
axes[1].legend()

plt.tight_layout()
plt.show()

# ============================================================
# %% [markdown]
# ## 3. Model Interpretation
#
# Untuk Logistic Regression, interpretasi terbaik adalah melalui
# KOEFISIEN MODEL — bukan feature importance (itu untuk tree-based)
# dan bukan SHAP TreeExplainer (itu untuk tree-based).
#
# Koefisien LR = log-odds. Nilai positif berarti fitur tersebut
# meningkatkan probabilitas churn; negatif berarti menurunkannya.
# exp(koefisien) = odds ratio -> lebih mudah diinterpretasi secara bisnis.
# ============================================================

# %% Koefisien Logistic Regression
coef = pd.Series(model.coef_[0], index=X_train.columns)
coef_sorted = coef.abs().sort_values(ascending=False).head(15)
top_features = coef_sorted.index
top_coef = coef[top_features].sort_values()

# Odds ratio (exp koefisien) -> lebih interpretable secara bisnis
odds_ratio = np.exp(coef[top_features]).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Koefisien (log-odds)
colors = ["#d62728" if v > 0 else "#1f77b4" for v in top_coef.values]
axes[0].barh(top_coef.index, top_coef.values, color=colors)
axes[0].axvline(0, color="gray", linewidth=0.8)
axes[0].set_title(f"Top 15 Koefisien — {MODEL_NAME}\n"
                   "(Merah = dorong Churn, Biru = dorong No Churn)")
axes[0].set_xlabel("Log-Odds Coefficient")

# Plot 2: Odds Ratio
or_sorted = odds_ratio.sort_values()
colors2 = ["#d62728" if v > 1 else "#1f77b4" for v in or_sorted.values]
axes[1].barh(or_sorted.index, or_sorted.values, color=colors2)
axes[1].axvline(1, color="gray", linewidth=0.8, linestyle="--")
axes[1].set_title("Odds Ratio per Fitur\n(>1 = meningkatkan risiko churn)")
axes[1].set_xlabel("Odds Ratio (exp(coefficient))")

plt.tight_layout()
plt.show()

# %% Tabel odds ratio (untuk laporan/README)
or_table = pd.DataFrame({
    "Feature":     odds_ratio.sort_values(ascending=False).index,
    "Coefficient": coef[odds_ratio.sort_values(ascending=False).index].round(4).values,
    "Odds_Ratio":  odds_ratio.sort_values(ascending=False).round(4).values,
    "Interpretation": [
        "meningkatkan risiko churn" if v > 1 else "menurunkan risiko churn"
        for v in odds_ratio.sort_values(ascending=False).values
    ]
})
print("\nTop 15 Fitur — Koefisien & Odds Ratio:")
print(or_table.to_string(index=False))

# %% [markdown]
# Cara membaca Odds Ratio:
# - OR > 1: fitur ini meningkatkan risiko churn. Contoh OR=2.5 artinya
#   customer dengan fitur ini 2.5x lebih mungkin churn dibanding baseline.
# - OR < 1: fitur ini menurunkan risiko churn. Contoh OR=0.3 artinya
#   customer dengan fitur ini hanya 0.3x (lebih kecil) kemungkinan churn.
# - OR = 1: fitur tidak berpengaruh pada probabilitas churn.

# ============================================================
# %% [markdown]
# ## 4. Threshold Tuning Summary
# ============================================================

rows = []
for t in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    pt  = (y_proba >= t).astype(int)
    cmt = confusion_matrix(y_test, pt)
    tnt, fpt, fnt, tpt = cmt.ravel()
    rec  = tpt / (tpt + fnt) if (tpt + fnt) > 0 else 0
    prec = tpt / (tpt + fpt) if (tpt + fpt) > 0 else 0
    f2   = fbeta_score(y_test, pt, beta=2)
    rows.append({
        "threshold": t,
        "recall": round(rec, 3),
        "precision": round(prec, 3),
        "f2": round(f2, 3),
        "FN": fnt,
        "FP": fpt,
        "selected": "← DIPILIH" if t == THRESHOLD else ""
    })

thr_table = pd.DataFrame(rows)
print("\nTrade-off di berbagai threshold:")
print(thr_table.to_string(index=False))

# ============================================================
# %% [markdown]
# ## 5. Business Translation
# ============================================================

# %% Risk segmentation
risk_df = pd.DataFrame({
    "actual_churn":    y_test.values,
    "churn_proba":     y_proba,
    "monthly_charges": X_test["MonthlyCharges"].values
})

def segment(p):
    if p >= 0.6:
        return "High Risk"
    elif p >= 0.35:
        return "Medium Risk"
    else:
        return "Low Risk"

risk_df["segment"] = risk_df["churn_proba"].apply(segment)

summary = risk_df.groupby("segment").agg(
    jumlah_customer   = ("actual_churn", "count"),
    actual_churn_rate = ("actual_churn", "mean"),
    avg_monthly_charges = ("monthly_charges", "mean")
).reindex(["High Risk", "Medium Risk", "Low Risk"]).round(3)

print("\nRisk Segmentation:")
print(summary)

# Validasi: churn rate harus monotonically decreasing High -> Medium -> Low
rates = summary["actual_churn_rate"].values
monotonic = all(rates[i] >= rates[i+1] for i in range(len(rates)-1)
                if not np.isnan(rates[i]) and not np.isnan(rates[i+1]))
print(f"\nMonotonic churn rate (validasi segmentasi): {'✓ Valid' if monotonic else '✗ Perlu review'}")

# %% Estimasi business impact
high_risk        = risk_df[risk_df["segment"] == "High Risk"]
high_risk_churn  = high_risk[high_risk["actual_churn"] == 1]

success_rate     = 0.30
n_saved          = len(high_risk_churn) * success_rate
monthly_saved    = n_saved * high_risk_churn["monthly_charges"].mean() if len(high_risk_churn) > 0 else 0
annual_saved     = monthly_saved * 12

print(f"\nBusiness Impact Estimation:")
print(f"Customer High Risk              : {len(high_risk)}")
print(f"...yang benar-benar churn       : {len(high_risk_churn)}")
print(f"Estimasi terselamatkan (30%)    : {n_saved:.1f} customer")
print(f"Revenue terselamatkan / bulan   : ${monthly_saved:,.2f}")
print(f"Revenue terselamatkan / tahun   : ${annual_saved:,.2f}")

# %% [markdown]
# ## Ringkasan Akhir Project
#
# Model Final : {MODEL_NAME}
# Threshold   : {THRESHOLD}
# ROC-AUC     : {metadata['roc_auc']}
# Recall      : {metadata['recall']:.1%} ({tp} dari {tp+fn} customer churn terdeteksi)
# False Neg   : {fn} customer churn tidak terdeteksi
# False Pos   : {fp} false alarm (dapat retention offer tidak perlu)
# Precision   : {metadata['precision']:.1%}
#
# Rekomendasi Bisnis:
# 1. Fokus retention campaign pada segmen High Risk (probabilitas >= 0.6)
# 2. Tawarkan migrasi kontrak month-to-month -> tahunan dengan insentif
#    (Contract_Two year adalah faktor protektif terkuat, odds ratio jauh < 1)
# 3. Program onboarding/loyalty untuk pelanggan baru (tenure rendah)
# 4. Evaluasi pricing/kualitas layanan Fiber optic
# 5. Dorong migrasi dari Electronic check ke pembayaran otomatis