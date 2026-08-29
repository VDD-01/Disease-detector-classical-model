import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "../data/heart_disease_uci.csv"

df = pd.read_csv(DATA_PATH)


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nTarget distribution (num):")
print(df["num"].value_counts().sort_index())


# ── Drop non-feature columns ──────────────────────────────────────
# 'id'      : row identifier, carries no predictive value
# 'dataset' : source hospital name, would cause data leakage

df = df.drop(columns=["id", "dataset"])


# ── Map boolean columns to integers ──────────────────────────────
# The CSV parser reads TRUE/FALSE as Python booleans (True/False),
# not strings. Missing rows show up as NaN.
# Impute NaNs with mode first (preserving the boolean type),
# then cast to int (0/1).

bool_cols = ["fbs", "exang"]

for col in bool_cols:
    col_mode = df[col].dropna().mode()
    if len(col_mode) > 0:
        df[col] = df[col].fillna(col_mode.iloc[0])
    else:
        df[col] = df[col].fillna(False)    # safe fallback
    df[col] = df[col].astype(int)


# ── Impute remaining missing values ──────────────────────────────
# Numeric columns            → median imputation
# Categorical string columns → mode imputation
#
# Missing value counts (from raw data):
#   trestbps : 59   chol : 30   thalch : 55   oldpeak : 62
#   ca : 611   restecg : 2   slope : 309   thal : 486

numeric_impute_cols = ["trestbps", "chol", "thalch", "oldpeak", "ca"]

for col in numeric_impute_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical string columns → mode imputation
for col in ["restecg", "slope", "thal"]:
    df[col] = df[col].fillna(df[col].mode().iloc[0])


# ── Binarize target ───────────────────────────────────────────────
# The original 'num' column uses 0–4 (severity of disease).
# Standard practice binarizes it:
#   0          → 0  (No disease)
#   1, 2, 3, 4 → 1  (Disease present)

df["target"] = (df["num"] >= 1).astype(int)

df = df.drop(columns=["num"])

print("\nBinarized target distribution:")
print(df["target"].value_counts().sort_index())


# ── One-hot encode categorical columns ───────────────────────────
# Columns: sex, cp, restecg, slope, thal

categorical_cols = ["sex", "cp", "restecg", "slope", "thal"]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Convert boolean dummy columns to int (0/1) for clean CSV output
bool_dummy_cols = df.select_dtypes(include="bool").columns
df[bool_dummy_cols] = df[bool_dummy_cols].astype(int)

print("\nDataset shape after encoding:", df.shape)
print("\nFeature columns:")
print([c for c in df.columns if c != "target"])


# ── Train / test split ────────────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape  :", y.shape)

# stratify=y preserves the disease/no-disease ratio in both splits.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())


# ── Save unscaled CSVs ────────────────────────────────────────────
# These are a clean checkpoint before scaling and are used
# by classical_tuning.py (which includes its own scaler in the pipeline).

train_df = X_train.copy()
train_df["target"] = y_train.values

test_df = X_test.copy()
test_df["target"] = y_test.values

train_df.to_csv("../train_features.csv", index=False)
test_df.to_csv("../test_features.csv",   index=False)

print("\nSaved: train_features.csv")
print("Saved: test_features.csv")


# ── StandardScaler ────────────────────────────────────────────────
# Fit ONLY on training data, then apply to test to prevent leakage.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns,
    index=X_test.index
)


print("\n" + "=" * 60)
print("SCALED DATA")
print("=" * 60)

print("Scaled training shape:", X_train_scaled.shape)
print("Scaled testing shape :", X_test_scaled.shape)

print("\nTraining feature means (should be close to 0):")
print(X_train_scaled.mean().round(4))

print("\nTraining feature std devs (should be close to 1):")
print(X_train_scaled.std().round(4))


train_scaled_df = X_train_scaled.copy()
train_scaled_df["target"] = y_train.values

test_scaled_df = X_test_scaled.copy()
test_scaled_df["target"] = y_test.values

train_scaled_df.to_csv("../train_features_scaled.csv", index=False)
test_scaled_df.to_csv("../test_features_scaled.csv",   index=False)

print("\nSaved: train_features_scaled.csv")
print("Saved: test_features_scaled.csv")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)
