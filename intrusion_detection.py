import streamlit as st
import joblib
import pandas as pd
import smtplib
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

USER_CREDENTIALS = {"admin": "admin","Cult":"Cult" ,"user": "securepass"}

# SMTP Email Configuration (Update with your details)
SMTP_SERVER = "smtp.gmail.com"  # Example: Gmail SMTP
SMTP_PORT = 587  # Standard port for TLS
SENDER_EMAIL = "anilreddy5452@gmail.com"
SENDER_PASSWORD = "tdvu xfzl ikuy txvv"
RECIPIENT_EMAIL = "anilreddy110154@gmail.com"  # Where alerts will be sent

def send_email_alert(attack_df):
    subject = "🚨 Intrusion Detection Alert: Attacks Detected"
    attack_info = attack_df.to_string(index=False)
    body = f"""
    ALERT! 🚨

    Intrusion detection system has detected {len(attack_df)} attack(s).

    Details:
    {attack_info}

    Please take necessary action immediately.
    """

    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECIPIENT_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()  # Secure the connection
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        server.quit()
        st.success("📧 Alert email sent successfully!")
    except Exception as e:
        st.error(f"❌ Failed to send email: {e}")

def login():
    st.title("🔐 User Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password", key="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("❌ Invalid username or password!")

def load_and_predict():
    st.title("Intrusion Detection System - Batch Prediction")
    st.write(f"Welcome, **{st.session_state['username']}** 👋")

    # File Upload Section
    uploaded_file = st.file_uploader("📂 Upload a CSV file for prediction", type=["csv"])
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
        
        model = load_model("deep_learning_model.h5")

        encoder = joblib.load("encoder.pkl")  

        expected_features = joblib.load("feature_names.pkl")  

        scaler = joblib.load("scaler.pkl")  

        test_data = pd.read_csv(uploaded_file)

        test_data = test_data[[col for col in expected_features if col in test_data.columns]]

        num_cols = test_data.select_dtypes(include=["float64", "int64"]).columns.tolist()
        cat_cols = ["protocol_type", "service", "flag"]

        for col in cat_cols:
            if col in test_data.columns:
                test_data[col] = test_data[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1)

        test_data[num_cols] = scaler.transform(test_data[num_cols])

        test_data_array = np.array(test_data)

        predictions = model.predict(test_data_array)

        predicted_classes = np.argmax(predictions, axis=1)

        label_mapping = {0: "Normal", 1: "Attack"}
        test_data["Prediction"] = [label_mapping[pred] for pred in predicted_classes]

        attack_df = test_data[test_data["Prediction"] == "Attack"]

        if attack_df.empty:
            st.success("✅ No attacks detected in the dataset!")
        else:
            attack_df = attack_df.head(100)
            st.subheader("🚨 Detected Attacks")
            st.dataframe(attack_df)

            send_email_alert(attack_df)

        csv = attack_df.to_csv(index=False)
        st.download_button("📥 Download Attack Data", csv, "attacks.csv", "text/csv")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
else:
    load_and_predict()
