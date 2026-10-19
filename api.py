from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from src.email_alert import send_lead_alert

app = FastAPI(title="Sales Automation AI")

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "sales_lead_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


class LeadRequest(BaseModel):
    Industry: str
    Company_Size: int
    Website_Visits: int
    Emails_Opened: int
    Demo: str
    Budget: float


@app.post("/predict")
def predict_lead(lead: LeadRequest):

    # Prepare input data
    data = pd.DataFrame([
        {
            "Industry": lead.Industry,
            "Company Size": lead.Company_Size,
            "Website Visits": lead.Website_Visits,
            "Emails Opened": lead.Emails_Opened,
            "Demo": lead.Demo,
            "Budget": lead.Budget
        }
    ])

    # Predict lead category
    prediction = model.predict(data)
    result = label_encoder.inverse_transform(prediction)[0]

    # Prepare lead details for the email
    lead_details = {
        "Industry": lead.Industry,
        "Company Size": lead.Company_Size,
        "Website Visits": lead.Website_Visits,
        "Emails Opened": lead.Emails_Opened,
        "Demo": lead.Demo,
        "Budget": lead.Budget
    }

    # Send email alert
    try:
        email_status = send_lead_alert(lead_details, result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction succeeded, but email sending failed: {str(e)}"
        )

    # Return prediction and email status
    return {
        "prediction": result,
        "email_status": email_status
    }
