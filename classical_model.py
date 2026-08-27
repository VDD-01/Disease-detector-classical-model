import pandas as pd
import time

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier



TRAIN_PATH = "train_30features_scaled.csv"
TEST_PATH = "test_30features_scaled.csv"

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]

X_test = test_data.drop(columns=["diagnosis"])
y_test = test_data["diagnosis"]


print("=" * 70)
print("DATASET")
print("=" * 70)

print("Training samples :", X_train.shape[0])
print("Testing samples  :", X_test.shape[0])
print("Number of features:", X_train.shape[1])



models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}


results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)


    start_time = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start_time



    y_pred = model.predict(X_test)

    # Probability of malignant class
    y_probability = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred
    ).ravel()


    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1
    )

    sensitivity = recall_score(
        y_test,
        y_pred,
        pos_label=1
    )

    specificity = recall_score(
        y_test,
        y_pred,
        pos_label=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability
    )



    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Malignant"]
        )
    )

    print("Accuracy    :", round(accuracy, 4))
    print("Sensitivity :", round(sensitivity, 4))
    print("Specificity :", round(specificity, 4))
    print("Precision   :", round(precision, 4))
    print("F1 Score    :", round(f1, 4))
    print("ROC-AUC     :", round(roc_auc, 4))
    print("Training Time:", round(training_time, 4), "seconds")



    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "Training Time (s)": training_time,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    })

results_df = pd.DataFrame(results)

print("FINAL MODEL COMPARISON")

print(
    results_df[
        [
            "Model",
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "Precision",
            "F1 Score",
            "ROC-AUC",
            "Training Time (s)"
        ]
    ].round(4).to_string(index=False)
)



results_df.to_csv(
    "classical_model_results.csv",
    index=False
)

print("\nResults saved to:")
print("classical_model_results.csv")


best_accuracy = results_df.loc[
    results_df["Accuracy"].idxmax()
]

best_sensitivity = results_df.loc[
    results_df["Sensitivity"].idxmax()
]

best_specificity = results_df.loc[
    results_df["Specificity"].idxmax()
]

best_auc = results_df.loc[
    results_df["ROC-AUC"].idxmax()
]


print("\n" + "=" * 70)
print("BEST PERFORMING MODELS")
print("=" * 70)

print(
    "Best Accuracy    :",
    best_accuracy["Model"],
    f"({best_accuracy['Accuracy']:.4f})"
)

print(
    "Best Sensitivity :",
    best_sensitivity["Model"],
    f"({best_sensitivity['Sensitivity']:.4f})"
)

print(
    "Best Specificity :",
    best_specificity["Model"],
    f"({best_specificity['Specificity']:.4f})"
)

print(
    "Best ROC-AUC     :",
    best_auc["Model"],
    f"({best_auc['ROC-AUC']:.4f})"
)