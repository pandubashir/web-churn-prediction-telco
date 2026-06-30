# %% [markdown]
# # 02 - Data Preparation
# Tujuan: membersihkan data, melakukan feature engineering, encoding,
# dan membagi data menjadi train-test set yang siap dipakai untuk modeling.

# %% Import library
import pandas as pd
import numpy as np

# %% Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
print("Shape awal:", df.shape)

# %% [markdown]
# ## 1. Cleaning
# - Drop customerID (tidak prediktif, hanya identifier)
# - Fix TotalCharges (object -> numeric)

# %% Drop customerID
df = df.drop(columns=["customerID"])

# %% Fix TotalCharges
# TotalCharges punya beberapa baris berisi string kosong " " (customer baru, tenure = 0)
# pd.to_numeric dengan errors="coerce" akan mengubah string kosong jadi NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

print("Jumlah NaN di TotalCharges setelah konversi:", df["TotalCharges"].isnull().sum())

# Untuk customer dengan tenure = 0 (baru join), TotalCharges yang masuk akal adalah 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("Jumlah NaN setelah fillna:", df["TotalCharges"].isnull().sum())

# %% [markdown]
# ## 2. Feature Engineering
# - tenure_group: binning tenure ke beberapa kategori
# - num_services: total jumlah layanan tambahan yang di-subscribe

# %% Buat tenure_group
def tenure_to_group(tenure):
    if tenure <= 12:
        return "0-12"
    elif tenure <= 24:
        return "13-24"
    elif tenure <= 48:
        return "25-48"
    elif tenure <= 60:
        return "49-60"
    else:
        return "61+"

df["tenure_group"] = df["tenure"].apply(tenure_to_group)

print(df["tenure_group"].value_counts())

# %% Buat num_services
# Hitung berapa banyak layanan tambahan (selain phone & internet dasar)
# yang di-subscribe oleh customer. Nilai "Yes" dihitung 1, selain itu 0.
service_cols = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

df["num_services"] = (df[service_cols] == "Yes").sum(axis=1)

print(df["num_services"].value_counts().sort_index())

# %% [markdown]
# ## 3. Encoding
# - Binary categorical (Yes/No, Male/Female, dst) -> 0/1
# - Multi-category categorical -> one-hot encoding
# - Target (Churn) -> 0/1

# %% Encode target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(df["Churn"].value_counts())

# %% Encode binary categorical columns
binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]

for col in binary_cols:
    print(f"{col}: {df[col].unique()}")

# gender: Male/Female -> 1/0
df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# kolom Yes/No lainnya -> 1/0
for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df[col] = df[col].map({"Yes": 1, "No": 0})

# %% One-hot encoding untuk kolom kategorikal sisanya
# drop_first=True untuk mengurangi redundansi (menghindari dummy variable trap)
categorical_cols = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod", "tenure_group"
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Shape setelah encoding:", df_encoded.shape)
df_encoded.head()

# %% [markdown]
# ## 4. Train-Test Split
# - 80:20 split
# - stratify=y agar proporsi churn/non-churn tetap konsisten di train & test

# %% Split fitur dan target
from sklearn.model_selection import train_test_split

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)
print("\nProporsi churn di train:")
print(y_train.value_counts(normalize=True).round(3))
print("\nProporsi churn di test:")
print(y_test.value_counts(normalize=True).round(3))

# %% [markdown]
# ## 5. Simpan hasil ke file (agar bisa langsung dipakai di script modeling)

# %% Save processed data
X_train.to_csv("data/X_train.csv", index=False)
X_test.to_csv("data/X_test.csv", index=False)
y_train.to_csv("data/y_train.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)

print("Data preparation selesai. File tersimpan di folder data/:")
print("- X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# %% [markdown]
# ## Ringkasan
# 1. customerID di-drop, TotalCharges dikonversi ke numerik (NaN -> 0 untuk tenure=0).
# 2. Fitur baru: tenure_group (binning) dan num_services (jumlah layanan tambahan).
# 3. Encoding: binary -> 0/1, multi-category -> one-hot (drop_first=True).
# 4. Train-test split 80:20 dengan stratify pada target.
# 5. Data siap pakai disimpan ke data/X_train.csv, X_test.csv, y_train.csv, y_test.csv.
#
# Lanjut ke: 03_modeling.py
