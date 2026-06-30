# %% [markdown]
# # 01 - Data Loading & Exploratory Data Analysis (EDA)
# Dataset: Telco Customer Churn (Kaggle/IBM)
# Tujuan: memahami struktur data, distribusi target, dan data quality issues
# sebelum masuk ke tahap data preparation & modeling.

# %% Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# %% Load dataset
# Sesuaikan path jika file CSV kamu berada di lokasi lain
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Jumlah baris dan kolom:", df.shape)
df.head()

# %% Info umum dataset
df.info()

# %% [markdown]
# Perhatikan kolom `TotalCharges` — biasanya terbaca sebagai object/string,
# padahal seharusnya numerik. Ini salah satu data quality issue yang akan
# kita perbaiki di script preparation.

# %% Cek missing value (eksplisit)
print("Missing value per kolom:")
print(df.isnull().sum())

# %% Cek kolom TotalCharges yang bermasalah
# TotalCharges sering berisi string kosong " " untuk customer baru (tenure = 0)
total_charges_blank = df[df["TotalCharges"] == " "]
print("Jumlah baris dengan TotalCharges kosong:", len(total_charges_blank))
total_charges_blank[["customerID", "tenure", "TotalCharges", "Churn"]]

# %% Distribusi target (Churn)
churn_counts = df["Churn"].value_counts()
churn_pct = df["Churn"].value_counts(normalize=True) * 100

print(churn_counts)
print("\nProporsi (%):")
print(churn_pct.round(2))

plt.figure(figsize=(5, 4))
sns.countplot(data=df, x="Churn", palette="Set2")
plt.title("Distribusi Target: Churn")
plt.show()

# %% [markdown]
# Catatan penting: dataset ini IMBALANCED.
# Kelas "No" (~73%) jauh lebih banyak dari "Yes" (~27%).
# Implikasi: jangan gunakan accuracy sebagai metric utama,
# dan perlu strategi khusus (class_weight / SMOTE / scale_pos_weight)
# saat modeling nanti.

# %% Distribusi fitur numerik utama
numeric_cols = ["tenure", "MonthlyCharges"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, col in zip(axes, numeric_cols):
    sns.histplot(data=df, x=col, hue="Churn", kde=True, ax=ax, multiple="stack")
    ax.set_title(f"Distribusi {col} berdasarkan Churn")
plt.tight_layout()
plt.show()

# %% [markdown]
# Insight yang biasanya muncul:
# - Customer dengan tenure rendah (pelanggan baru) cenderung lebih banyak churn.
# - Customer dengan MonthlyCharges tinggi juga punya proporsi churn lebih tinggi.

# %% Hubungan Contract type dengan Churn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Churn berdasarkan Tipe Kontrak")
plt.xticks(rotation=15)
plt.show()

# %% [markdown]
# Insight yang biasanya muncul:
# - Customer dengan kontrak "Month-to-month" punya churn rate jauh lebih tinggi
#   dibanding "One year" atau "Two year".
# Insight ini akan jadi salah satu poin rekomendasi bisnis di akhir project.

# %% Cek kolom-kolom kategorikal dengan nilai "No internet service" / "No phone service"
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

for col in service_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

# %% [markdown]
# Catatan: nilai "No internet service" pada kolom-kolom di atas BUKAN missing value,
# melainkan kategori valid (customer tidak subscribe internet sama sekali).
# Ini perlu diperhatikan saat encoding di tahap data preparation.

# %% [markdown]
# ## Ringkasan EDA
# 1. Dataset terdiri dari ~7.043 baris, 21 kolom, 1 target (Churn).
# 2. Target imbalanced (~27% churn) -> perlu handling khusus saat modeling.
# 3. Kolom TotalCharges perlu dikonversi ke numerik (11 baris bermasalah).
# 4. Kolom customerID akan di-drop (tidak prediktif).
# 5. Tenure rendah, MonthlyCharges tinggi, dan Contract month-to-month
#    berasosiasi dengan churn rate lebih tinggi -> kandidat fitur penting.
#
# Lanjut ke: 02_data_preparation.py
