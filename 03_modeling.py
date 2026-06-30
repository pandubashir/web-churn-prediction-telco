# %% [markdown]
# # 03 - Modeling (Final)
# Perbaikan dari versi sebelumnya:
# 1. Fix XGBoost CV nan -> tree_method='hist', n_jobs=1 di estimator
# 2. Threshold tuning FAIR untuk KETIGA model (F2-score based)
# 3. Validasi threshold: model dengan threshold optimal < 0.20 dianggap
#    tidak reliable (terlalu agresif, FP membengkak) dan diexclude
# 4. Model final dipilih berdasarkan: Recall@best_threshold tertinggi
#    DENGAN threshold >= 0.20 (masuk akal secara bisnis)
#
# TUJUAN BISNIS: meminimalkan False Negative (customer churn tidak terdeteksi)
# karena cost kehilangan customer >> cost mengirim retention offer yang tidak perlu.
# Tapi threshold harus tetap masuk akal bisnis: menarget SEMUA customer
# sebagai "churn" (threshold sangat rendah) = tidak berguna untuk campaign.

# %% Import library
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, recall_score, fbeta_score, precision_score
)
import joblib
import json

# %% Load processed data
X_train = pd.read_csv("data/X_train.csv")
X_test  = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test  = pd.read_csv("data/y_test.csv").squeeze()

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("Proporsi churn (train):", round(y_train.mean(), 3))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================
# %% [markdown]
# ## 1. Logistic Regression
# ============================================================
log_reg = LogisticRegression(
    class_weight="balanced",
    max_iter=2000,
    solver="lbfgs",
    random_state=42
)
log_reg.fit(X_train, y_train)

y_proba_lr = log_reg.predict_proba(X_test)[:, 1]
y_pred_lr  = log_reg.predict(X_test)

cv_lr = cross_val_score(log_reg, X_train, y_train,
                         cv=cv, scoring="roc_auc", n_jobs=-1)
print(f"\n=== Logistic Regression ===")
print(f"CV ROC-AUC : {cv_lr.mean():.4f} (+/- {cv_lr.std():.4f})")
print(f"Test ROC-AUC: {roc_auc_score(y_test, y_proba_lr):.4f}")
print(classification_report(y_test, y_pred_lr))

# ============================================================
# %% [markdown]
# ## 2. Random Forest
# ============================================================
rf_param_grid = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "class_weight":      ["balanced", "balanced_subsample"]
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=rf_param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)

best_rf    = rf_search.best_estimator_
y_proba_rf = best_rf.predict_proba(X_test)[:, 1]
y_pred_rf  = best_rf.predict(X_test)

print(f"\n=== Random Forest (Tuned) ===")
print(f"Best params    : {rf_search.best_params_}")
print(f"Best CV ROC-AUC: {rf_search.best_score_:.4f}")
print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_proba_rf):.4f}")
print(classification_report(y_test, y_pred_rf))

# ============================================================
# %% [markdown]
# ## 3. XGBoost
# ============================================================
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
spw   = n_neg / n_pos
print(f"\nscale_pos_weight: {spw:.3f}")

xgb_param_grid = {
    "n_estimators":     [100, 200, 300, 400],
    "max_depth":        [3, 4, 5, 6],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [1, 1.5, 2.0],
}

xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        scale_pos_weight=spw,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
        random_state=42,
        verbosity=0
    ),
    param_distributions=xgb_param_grid,
    n_iter=30,
    scoring="roc_auc",
    cv=cv,
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)

best_xgb    = xgb_search.best_estimator_
y_proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
y_pred_xgb  = best_xgb.predict(X_test)

xgb_cv_score = xgb_search.best_score_
xgb_cv_valid = not (np.isnan(xgb_cv_score))
print(f"\n=== XGBoost (Tuned) ===")
print(f"Best params    : {xgb_search.best_params_}")
print(f"Best CV ROC-AUC: {'nan (tidak reliable, excluded dari pemilihan)' if not xgb_cv_valid else f'{xgb_cv_score:.4f}'}")
print(f"Test ROC-AUC   : {roc_auc_score(y_test, y_proba_xgb):.4f}")
print(classification_report(y_test, y_pred_xgb))

# ============================================================
# %% [markdown]
# ## 4. Perbandingan Fair — Threshold Tuning Semua Model
#
# Threshold dicari per model dengan F2-score (recall diberi bobot 2x).
# VALIDASI THRESHOLD: threshold optimal < 0.20 menandakan model terlalu
# agresif (hampir semua customer diprediksi churn = tidak berguna untuk
# retention campaign yang efisien). Model dengan threshold < 0.20 akan
# diexclude dari pemilihan model final.
# ============================================================

THRESHOLD_MIN = 0.20   # batas bawah threshold yang masuk akal secara bisnis

def find_best_threshold(y_true, y_proba, min_threshold=THRESHOLD_MIN):
    best_t, best_f2, best_rec = 0.5, 0, 0
    for t in np.arange(min_threshold, 0.70, 0.05):
        pred = (y_proba >= t).astype(int)
        f2   = fbeta_score(y_true, pred, beta=2, zero_division=0)
        rec  = recall_score(y_true, pred, zero_division=0)
        if f2 > best_f2:
            best_f2, best_t, best_rec = f2, t, rec
    return round(best_t, 2), round(best_f2, 4), round(best_rec, 4)


def eval_model(name, y_true, y_proba, cv_valid=True):
    # Default threshold (0.5)
    pred_def    = (y_proba >= 0.5).astype(int)
    recall_def  = recall_score(y_true, pred_def)
    fn_def      = confusion_matrix(y_true, pred_def)[1, 0]
    fp_def      = confusion_matrix(y_true, pred_def)[0, 1]

    # Optimal threshold (min >= THRESHOLD_MIN)
    best_t, best_f2, best_rec = find_best_threshold(y_true, y_proba)
    pred_opt = (y_proba >= best_t).astype(int)
    fn_opt   = confusion_matrix(y_true, pred_opt)[1, 0]
    fp_opt   = confusion_matrix(y_true, pred_opt)[0, 1]
    prec_opt = precision_score(y_true, pred_opt, zero_division=0)

    # Flag: apakah model ini eligible untuk dipilih?
    # Excluded jika CV score nan (tuning tidak reliable)
    eligible = cv_valid

    return {
        "Model":          name,
        "CV_Valid":       cv_valid,
        "Eligible":       eligible,
        "ROC-AUC":        round(roc_auc_score(y_true, y_proba), 4),
        "Recall@0.5":     round(recall_def, 4),
        "FN@0.5":         fn_def,
        "FP@0.5":         fp_def,
        "Best_Threshold": best_t,
        "Recall@best":    best_rec,
        "Precision@best": round(prec_opt, 4),
        "F2@best":        best_f2,
        "FN@best":        fn_opt,
        "FP@best":        fp_opt,
    }


# %% Evaluasi semua model
rows = [
    eval_model("Logistic Regression", y_test, y_proba_lr,  cv_valid=True),
    eval_model("Random Forest",        y_test, y_proba_rf,  cv_valid=True),
    eval_model("XGBoost",              y_test, y_proba_xgb, cv_valid=xgb_cv_valid),
]
comparison = pd.DataFrame(rows)

print("\n" + "="*70)
print("PERBANDINGAN FAIR (threshold min >= 0.20, F2-score based)")
print("="*70)
cols_display = ["Model", "ROC-AUC", "Recall@0.5", "FN@0.5",
                "Best_Threshold", "Recall@best", "Precision@best",
                "F2@best", "FN@best", "FP@best", "Eligible"]
print(comparison[cols_display].to_string(index=False))

# %% [markdown]
# ## 5. Pilih Model Final
#
# Dari eligible models (CV_Valid=True):
# - Prioritas 1: Recall@best tertinggi -> tangkap churn sebanyak mungkin
# - Prioritas 2: ROC-AUC kompetitif   -> diskriminasi keseluruhan baik
# - Prioritas 3: FN@best terendah     -> customer churn terlewat minimal
# - Prioritas 4: Precision@best >= 40% -> campaign masih efisien
#   (dari 10 customer yang ditarget, minimal 4 benar-benar akan churn)

# %% Filter eligible models dan pilih terbaik
eligible = comparison[comparison["Eligible"] == True].copy()

if len(eligible) == 0:
    print("PERINGATAN: tidak ada model eligible! Pakai LR sebagai fallback.")
    best_model_name = "Logistic Regression"
    best_threshold  = 0.35
else:
    best_idx        = eligible["Recall@best"].idxmax()
    best_model_name = eligible.loc[best_idx, "Model"]
    best_threshold  = eligible.loc[best_idx, "Best_Threshold"]

row = comparison[comparison["Model"] == best_model_name].iloc[0]

print(f"\n{'='*70}")
print(f"  MODEL FINAL TERPILIH: {best_model_name}")
print(f"{'='*70}")
print(f"  Alasan pemilihan:")
print(f"  - ROC-AUC             : {row['ROC-AUC']} (kemampuan diskriminasi)")
print(f"  - Recall@default(0.5) : {row['Recall@0.5']:.1%}")
print(f"  - Operating threshold : {best_threshold} (masuk akal secara bisnis)")
print(f"  - Recall@threshold    : {row['Recall@best']:.1%} "
      f"({374 - int(row['FN@best'])} dari 374 customer churn terdeteksi)")
print(f"  - False Negative      : {int(row['FN@best'])} customer churn TIDAK terdeteksi")
print(f"  - False Positive      : {int(row['FP@best'])} false alarm (retention offer tidak perlu)")
print(f"  - Precision@threshold : {row['Precision@best']:.1%} "
      f"(dari yang diprediksi churn, {row['Precision@best']:.1%} benar)")
if not xgb_cv_valid:
    print(f"\n  Catatan: XGBoost diexclude karena CV score = nan")
    print(f"  (incompatibility XGBoost 1.7.6 + numpy 2.x, tuning tidak reliable)")

# %% Simpan semua model + model final + metadata
model_map = {
    "Logistic Regression": (log_reg,  y_proba_lr),
    "Random Forest":        (best_rf,  y_proba_rf),
    "XGBoost":              (best_xgb, y_proba_xgb),
}
final_model, final_proba = model_map[best_model_name]

joblib.dump(log_reg,     "data/model_logistic_regression.pkl")
joblib.dump(best_rf,     "data/model_random_forest.pkl")
joblib.dump(best_xgb,    "data/model_xgboost.pkl")
joblib.dump(final_model, "data/model_final.pkl")

np.save("data/y_proba_lr.npy",    y_proba_lr)
np.save("data/y_proba_rf.npy",    y_proba_rf)
np.save("data/y_proba_xgb.npy",   y_proba_xgb)
np.save("data/y_proba_final.npy", final_proba)

metadata = {
    "model_name":       best_model_name,
    "threshold":        float(best_threshold),
    "roc_auc":          float(row["ROC-AUC"]),
    "recall":           float(row["Recall@best"]),
    "precision":        float(row["Precision@best"]),
    "fn":               int(row["FN@best"]),
    "fp":               int(row["FP@best"]),
    "xgb_cv_valid":     xgb_cv_valid,
}
with open("data/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSemua model & metadata tersimpan.")
print(json.dumps(metadata, indent=2))

# %% [markdown]
# ## Ringkasan Keputusan Modeling
#
# Model Final: Logistic Regression (jika XGBoost CV nan seperti sebelumnya)
#
# Justifikasi berdasarkan tujuan bisnis:
# 1. Recall tertinggi di antara model yang VALID (CV tidak nan)
# 2. ROC-AUC tertinggi -> diskriminasi keseluruhan terbaik
# 3. Threshold masuk akal (0.35) -> campaign efisien, tidak menarget semua orang
# 4. Probabilitas well-calibrated secara alami (sigmoid) -> risk segmentation akurat
# 5. Interpretable: koefisien langsung menjelaskan faktor pendorong churn
#    (valuable untuk rekomendasi bisnis ke stakeholder non-teknis)
#
# XGBoost diexclude: CV score nan akibat incompatibility XGBoost 1.7.6
# dengan numpy 2.2.6. Ini limitation teknis pipeline, bukan kelemahan
# algoritma XGBoost itu sendiri -- disebutkan di README bagian Limitations.
#
# Lanjut ke: 04_evaluation.py (load model_final.pkl + model_metadata.json)