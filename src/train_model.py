import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier


# =========================================================
# 1. Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "lead_training_data_100.csv"

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "sales_lead_model.pkl"

LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


# =========================================================
# 2. Load dataset
# =========================================================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# =========================================================
# 3. Separate features and target
# =========================================================

X = df.drop(columns=["Lead", "Result"])

y = df["Result"]


# =========================================================
# 4. Encode target
# =========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

print("\nClass mapping:")

for i, class_name in enumerate(label_encoder.classes_):
    print(i, "=", class_name)


# =========================================================
# 5. Define features
# =========================================================

numerical_features = [
    "Company Size",
    "Website Visits",
    "Emails Opened",
    "Budget"
]

categorical_features = [
    "Industry",
    "Demo"
]


# =========================================================
# 6. Train/Test Split
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# =========================================================
# 7. Preprocessing
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# =========================================================
# 8. XGBoost model
# =========================================================

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42
)


# =========================================================
# 9. Create Pipeline
# =========================================================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", xgb_model)
    ]
)


# =========================================================
# 10. Train
# =========================================================

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

print("Training completed!")


# =========================================================
# 11. Prediction
# =========================================================

y_pred = model.predict(X_test)


# =========================================================
# 12. Evaluation
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL EVALUATION")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# =========================================================
# 13. Create models directory
# =========================================================

MODELS_DIR.mkdir(exist_ok=True)


# =========================================================
# 14. Save XGBoost Pipeline
# =========================================================

joblib.dump(model, MODEL_PATH)


# =========================================================
# 15. Save LabelEncoder
# =========================================================

joblib.dump(label_encoder, LABEL_ENCODER_PATH)


# =========================================================
# 16. Confirm saving
# =========================================================

print("\n==============================")
print("MODEL SAVING")
print("==============================")

print("\nModel saved successfully!")
print("Model:", MODEL_PATH)

print("\nLabelEncoder saved successfully!")
print("LabelEncoder:", LABEL_ENCODER_PATH)