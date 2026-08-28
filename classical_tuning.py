import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV
)

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
    classification_report,
    make_scorer
)

from xgboost import XGBClassifier


TRAIN_PATH = "train_30features.csv"
TEST_PATH = "test_30features.csv"

RANDOM_STATE = 42
CV_FOLDS = 5


print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

X_train = train_data.drop(columns=["diagnosis"])
y_train = train_data["diagnosis"]

X_test = test_data.drop(columns=["diagnosis"])
y_test = test_data["diagnosis"]

print("Training samples :", len(X_train))
print("Testing samples  :", len(X_test))
print("Features         :", X_train.shape[1])


cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Sensitivity = recall of malignant class (1)
#
# Specificity = recall of benign class (0)

def specificity_score(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    if (tn + fp) == 0:
        return 0

    return tn / (tn + fp)


scoring = {
    "accuracy": "accuracy",
    "sensitivity": make_scorer(
        recall_score,
        pos_label=1
    ),
    "specificity": make_scorer(
        specificity_score
    ),
    "f1": make_scorer(
        f1_score,
        pos_label=1
    ),
    "roc_auc": "roc_auc"
}

models = {

    "Logistic Regression": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000,
                random_state=RANDOM_STATE
            ))
        ]),

        "parameters": {

            "model__C": [
                0.01,
                0.1,
                1,
                10,
                100
            ],

            "model__solver": [
                "liblinear",
                "lbfgs"
            ]
        }
    },


    "SVM": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                random_state=RANDOM_STATE
            ))
        ]),

        "parameters": {

            "model__C": [
                0.1,
                1,
                10,
                100
            ],

            "model__gamma": [
                "scale",
                0.001,
                0.01,
                0.1
            ],

            "model__kernel": [
                "rbf"
            ]
        }
    },


    "Random Forest": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced"
            ))
        ]),

        "parameters": {

            "model__n_estimators": [
                100,
                300,
                500
            ],

            "model__max_depth": [
                None,
                5,
                10,
                20
            ],

            "model__min_samples_split": [
                2,
                5,
                10
            ],

            "model__min_samples_leaf": [
                1,
                2,
                4
            ]
        }
    },


    "XGBoost": {

        "pipeline": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ))
        ]),

        "parameters": {

            "model__n_estimators": [
                100,
                200,
                300
            ],

            "model__max_depth": [
                2,
                3,
                4,
                5
            ],

            "model__learning_rate": [
                0.01,
                0.05,
                0.1
            ],

            "model__subsample": [
                0.8,
                1.0
            ],

            "model__colsample_bytree": [
                0.8,
                1.0
            ]
        }
    }
}

tuned_models = {}
cv_results = []

print("\n")
print("=" * 70)
print("HYPERPARAMETER TUNING + 5-FOLD CROSS-VALIDATION")
print("=" * 70)


for model_name, config in models.items():

    print("\n" + "-" * 70)
    print("Training:", model_name)
    print("-" * 70)

    start_time = time.time()

    grid_search = GridSearchCV(
        estimator=config["pipeline"],
        param_grid=config["parameters"],
        scoring=scoring,
        refit="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=False
    )

    grid_search.fit(
        X_train,
        y_train
    )

    elapsed_time = time.time() - start_time

    best_index = grid_search.best_index_

    results = grid_search.cv_results_

    tuned_models[model_name] = grid_search.best_estimator_

    print("\nBest parameters:")
    print(grid_search.best_params_)

    print(
        "\nBest CV ROC-AUC:",
        round(grid_search.best_score_, 4)
    )

    print(
        "Training/search time:",
        round(elapsed_time, 2),
        "seconds"
    )

    cv_results.append({

        "Model": model_name,

        "CV Accuracy":
            results["mean_test_accuracy"][best_index],

        "CV Accuracy Std":
            results["std_test_accuracy"][best_index],

        "CV Sensitivity":
            results["mean_test_sensitivity"][best_index],

        "CV Sensitivity Std":
            results["std_test_sensitivity"][best_index],

        "CV Specificity":
            results["mean_test_specificity"][best_index],

        "CV Specificity Std":
            results["std_test_specificity"][best_index],

        "CV F1":
            results["mean_test_f1"][best_index],

        "CV F1 Std":
            results["std_test_f1"][best_index],

        "CV ROC-AUC":
            results["mean_test_roc_auc"][best_index],

        "CV ROC-AUC Std":
            results["std_test_roc_auc"][best_index],

        "Search Time (s)":
            elapsed_time
    })


cv_results_df = pd.DataFrame(cv_results)

print("\n\n")
print("=" * 70)
print("CROSS-VALIDATION RESULTS")
print("=" * 70)

display_columns = [
    "Model",
    "CV Accuracy",
    "CV Sensitivity",
    "CV Specificity",
    "CV F1",
    "CV ROC-AUC"
]

print(
    cv_results_df[display_columns]
    .round(4)
    .to_string(index=False)
)

# We select the model based on cross-validated ROC-AUC.
#
# ROC-AUC is preferable to simply selecting based on
# test-set accuracy because the test set must remain
# untouched until final evaluation.

best_model_name = cv_results_df.loc[
    cv_results_df["CV ROC-AUC"].idxmax(),
    "Model"
]

best_model = tuned_models[best_model_name]


print("\n")
print("=" * 70)
print("SELECTED MODEL")
print("=" * 70)

print("Model:", best_model_name)

print(
    "CV ROC-AUC:",
    round(
        cv_results_df.loc[
            cv_results_df["Model"] == best_model_name,
            "CV ROC-AUC"
        ].iloc[0],
        4
    )
)


print("\n")
print("=" * 70)
print("FINAL TEST-SET EVALUATION")
print("=" * 70)

# The test set has not been used during model selection.
# This is the first time the selected model sees X_test.

start_time = time.time()

y_pred = best_model.predict(X_test)

prediction_time = time.time() - start_time


y_probability = best_model.predict_proba(
    X_test
)[:, 1]


cm = confusion_matrix(
    y_test,
    y_pred
)

tn, fp, fn, tp = cm.ravel()


print("\nConfusion Matrix:")
print(cm)

print("\n")
print("True Negatives :", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives :", tp)


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


print("\nFinal Test Metrics:")
print("----------------------------")

print("Accuracy    :", round(accuracy, 4))
print("Sensitivity :", round(sensitivity, 4))
print("Specificity :", round(specificity, 4))
print("Precision   :", round(precision, 4))
print("F1 Score    :", round(f1, 4))
print("ROC-AUC     :", round(roc_auc, 4))

print(
    "Prediction time:",
    round(prediction_time, 6),
    "seconds"
)


print("\n")
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Benign",
            "Malignant"
        ]
    )
)


cv_results_df.to_csv(
    "classical_cv_results.csv",
    index=False
)


final_results = pd.DataFrame([{

    "Model": best_model_name,

    "Accuracy": accuracy,

    "Sensitivity": sensitivity,

    "Specificity": specificity,

    "Precision": precision,

    "F1 Score": f1,

    "ROC-AUC": roc_auc,

    "Prediction Time (s)": prediction_time,

    "True Negatives": tn,

    "False Positives": fp,

    "False Negatives": fn,

    "True Positives": tp

}])

final_results.to_csv(
    "final_classical_model_results.csv",
    index=False
)

# Because the pipeline contains BOTH:
#
# StandardScaler
#       +
# Final model
#
# we can save them together.

joblib.dump(
    best_model,
    "breast_cancer_classical_model.joblib"
)


feature_info = pd.DataFrame({
    "Feature": X_train.columns
})

feature_info.to_csv(
    "model_features.csv",
    index=False
)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Benign", "Malignant"]
)

plt.yticks(
    [0, 1],
    ["Benign", "Malignant"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    "final_confusion_matrix.png",
    dpi=300
)

plt.show()


print("\n")
print("=" * 70)
print("TRAINING PIPELINE COMPLETE")
print("=" * 70)

print("Selected model :", best_model_name)
print("Features       :", X_train.shape[1])
print("Training data  :", X_train.shape[0])
print("Testing data   :", X_test.shape[0])

print("\nFinal test performance:")
print("Accuracy       :", f"{accuracy:.4f}")
print("Sensitivity    :", f"{sensitivity:.4f}")
print("Specificity    :", f"{specificity:.4f}")
print("F1 Score       :", f"{f1:.4f}")
print("ROC-AUC        :", f"{roc_auc:.4f}")

print("\nFiles created:")
print("1. classical_cv_results.csv")
print("2. final_classical_model_results.csv")
print("3. breast_cancer_classical_model.joblib")
print("4. model_features.csv")
print("5. final_confusion_matrix.png")