import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

GMAIL_ADDRESS = os.getenv("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
SALES_TEAM_EMAIL = os.getenv("SALES_TEAM_EMAIL")


def send_lead_alert(lead, prediction):

    priority = {
        "HOT": "HIGH",
        "WARM": "MEDIUM",
        "COLD": "LOW"
    }

    subject = f"{prediction} Lead Alert - {priority[prediction]} Priority"

    body = f"""
Sales Automation AI - Lead Alert

Industry: {lead['Industry']}
Company Size: {lead['Company Size']}
Website Visits: {lead['Website Visits']}
Emails Opened: {lead['Emails Opened']}
Demo: {lead['Demo']}
Budget: {lead['Budget']}

Predicted Category: {prediction}
Priority: {priority[prediction]}

Recommended Action:
"""

    if prediction == "HOT":
        body += "Contact this lead as soon as possible."
    elif prediction == "WARM":
        body += "Schedule a follow-up with this lead."
    else:
        body += "Add this lead to the nurture list."

    message = EmailMessage()
    message["From"] = GMAIL_ADDRESS
    message["To"] = SALES_TEAM_EMAIL
    message["Subject"] = subject
    message.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.send_message(message)

    return "Email alert sent successfully!"