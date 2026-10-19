import joblib

from pathlib import Path
import pandas as pd


# =========================================================
# 1. Project paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "sales_lead_model.pkl"

LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"


# =========================================================
# 2. Load model
# =========================================================

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# =========================================================
# 3. Load LabelEncoder
# =========================================================

label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("LabelEncoder loaded successfully!")


# =========================================================
# 4. Create new lead
# =========================================================

new_lead = pd.DataFrame([
    {
        "Industry": "SaaS",
        "Company Size": 100,
        "Website Visits": 25,
        "Emails Opened": 8,
        "Demo": "Yes",
        "Budget": 100000
    }
])


# =========================================================
# 5. Predict
# =========================================================

prediction = model.predict(new_lead)


# =========================================================
# 6. Convert numerical prediction to actual label
# =========================================================

predicted_class = label_encoder.inverse_transform(prediction)

result = predicted_class[0]


# =========================================================
# 7. Display result
# =========================================================

print("\n==============================")
print("LEAD PREDICTION")
print("==============================")

print("Industry:", new_lead["Industry"].iloc[0])
print("Company Size:", new_lead["Company Size"].iloc[0])
print("Website Visits:", new_lead["Website Visits"].iloc[0])
print("Emails Opened:", new_lead["Emails Opened"].iloc[0])
print("Demo:", new_lead["Demo"].iloc[0])
print("Budget:", new_lead["Budget"].iloc[0])

print("\nPredicted Lead:", result)