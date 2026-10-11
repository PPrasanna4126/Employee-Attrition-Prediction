import json
import os
import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/expanded_employee_data.csv")

features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "DistanceFromHome",
    "Department",
    "OverTime",
]

X = df[features].copy()

y = df["Attrition"].map({"Yes": 1, "No": 0})

if y.isna().any():
    raise ValueError(
        "The Attrition column contains missing or unexpected values."
    )

numeric_features = [
    "Age",
    "MonthlyIncome",
    "YearsAtCompany",
    "JobSatisfaction",
    "WorkLifeBalance",
    "PerformanceRating",
    "DistanceFromHome",
]

categorical_features = ["Department", "OverTime"]

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ),
}

results = []
trained_models = {}

for name, classifier in models.items():
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)
    trained_models[name] = pipeline

    print(f"\n{name}")
    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

results_df = pd.DataFrame(results)
results_df.to_csv("models/model_comparison.csv", index=False)

best_model_name = results_df.loc[results_df["F1"].idxmax(), "Model"]

best_model = trained_models[best_model_name]

joblib.dump(best_model, "models/best_model.pkl")

with open("models/best_model_info.json", "w") as file:
    json.dump({"best_model": best_model_name}, file, indent=4)

print("\nBest model:", best_model_name)
print("Model saved to models/best_model.pkl")
print("Metrics saved to models/model_comparison.csv")