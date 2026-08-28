import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DATA_PATH = "data/wdbc.data"

df = pd.read_csv(
    DATA_PATH,
    header=None
)



features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave_points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",

    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave_points_se",
    "symmetry_se",
    "fractal_dimension_se",

    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave_points_worst",
    "symmetry_worst",
    "fractal_dimension_worst"
]

columns = ["id", "diagnosis"] + features

df.columns = columns


print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nDuplicate IDs:", df["id"].duplicated().sum())

print("\nDiagnosis distribution:")
print(df["diagnosis"].value_counts())


df = df.drop(columns=["id"])


# B = Benign = 1
# M = Malignant = 0

df["diagnosis"] = df["diagnosis"].map({
    "B": 1,
    "M": 0
})


# Verify encoding
print("\nEncoded diagnosis distribution:")
print(df["diagnosis"].value_counts())


X = df.drop(columns=["diagnosis"])
y = df["diagnosis"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# random_state=42 ensures reproducibility.
# stratify=y preserves the benign/malignant ratio.

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
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# These contain the original feature values.
# They are useful as a clean checkpoint before scaling.

train_30 = X_train.copy()
train_30["diagnosis"] = y_train

test_30 = X_test.copy()
test_30["diagnosis"] = y_test

train_30.to_csv(
    "train_30features.csv",
    index=False
)

test_30.to_csv(
    "test_30features.csv",
    index=False
)


# StandardScaler transforms each feature so that the
# training data has approximately:
#
# mean = 0
# standard deviation = 1
#
# IMPORTANT:
# The scaler is FIT ONLY on training data.
# The same scaler is then used to transform test data.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


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

print("\nTraining feature standard deviations (should be close to 1):")
print(X_train_scaled.std().round(4))

train_30_scaled = X_train_scaled.copy()
train_30_scaled["diagnosis"] = y_train

test_30_scaled = X_test_scaled.copy()
test_30_scaled["diagnosis"] = y_test

train_30_scaled.to_csv(
    "train_30features_scaled.csv",
    index=False
)

test_30_scaled.to_csv(
    "test_30features_scaled.csv",
    index=False
)