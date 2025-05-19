import streamlit as st
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from twilio.rest import Client
from streamlit_autorefresh import st_autorefresh
import datetime

# Initialize Firebase only once
if not firebase_admin._apps:
    cred = credentials.Certificate("credentials.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://smartsensor-data-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })


def send_sms_alert(to_number, message):
    account_sid = 'AC3411cd714d34f4e3efd3a94bb2b0ec86'
    auth_token = 'b1661132829372d7060b54bf9a4d8476'
    twilio_number = '+19134233487'

    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            body=message,
            from_=twilio_number,
            to=to_number
        )
        print(f"SMS sent: {message.sid}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")

# Load models
xgb_model = joblib.load('fire_risk_xgb.pkl')
dt_model = joblib.load('decision_tree_model.pkl')

st.title("🌲 Forest Fire Risk Dashboard")

st_autorefresh(interval=10000, key="auto_refresh")


current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.info(f"⏱️ Current Time: {current_time}")

# Fetch sensor data
sensor_ref = db.reference('/sensor')
sensor_data = sensor_ref.get()

# Fetch previous DMC and DC (initialize if not present)
indices_ref = db.reference('/fire_indices')
indices_data = indices_ref.get() or {}

# Hardcoded values for initial test
previous_dmc = 6
previous_dc = 15

if sensor_data:
    st.subheader("📡 Live Input Data from Firebase:")
    st.json(sensor_data)

    # Raw values
    temp_c = sensor_data['tempDHT']
    humidity = sensor_data['humidity']
    pressure = sensor_data['pressure']
    soil_moisture = sensor_data['soil']
    smoke_density = sensor_data['mq7']

    # Derived values
    temp_k = temp_c + 273.15
    air_density = pressure * 100 / (287.05 * temp_k)
    wind_speed = 15

    st.write("🌬️ Air Density:", air_density)
    st.write("💨 Wind Speed:", wind_speed)

    # Fire weather indices
    ffmc = (59.5 * (1 - (humidity / 100))) + (temp_c - 10) * 0.25 + wind_speed * 0.5
    k = (0.36 * (temp_c + 2)) * (1 - (humidity / 100.0)) * (12 / 30.0)
    dmc = previous_dmc + k
    dc = previous_dc + 0.36 * (temp_c + 2.8)
    isi = 0.28 * np.exp(0.05039 * ffmc) * (1 + (np.power(wind_speed, 1.5) / 100))

    if dmc <= 0.4 * dc:
        bui = (0.8 * dmc * dc) / (dmc + 0.4 * dc)
    else:
        bui = dmc - (1 - (0.8 * dc) / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)

    if bui <= 80:
        ab = 0.626 * (bui ** 0.809) + 2
    else:
        ab = 1000 / (25 + 108.64 * np.exp(-0.023 * bui))

    B = 0.1 * isi * ab
    fwi = 11  # You can use formula if you prefer dynamic FWI

    st.subheader("🧮 Calculated Fire Weather Indices")
    st.write("🔥 FFMC:", ffmc)
    st.write("🔥 DMC:", dmc)
    st.write("🔥 DC:", dc)
    st.write("🔥 ISI:", isi)
    st.write("🔥 BUI:", bui)
    st.write("🔥 FWI:", fwi)
    st.write("📈 K (DMC factor):", k)

    # Round values
    ffmc = round(max(ffmc, 0), 2)
    dmc = round(max(dmc, 0), 2)
    dc = round(max(dc, 0), 2)
    isi = round(max(isi, 0), 2)
    bui = round(max(bui, 0), 2)
    fwi = round(max(fwi, 0), 2)

    # Input array for model
    input_values = np.array([[temp_c, temp_k, humidity, pressure, soil_moisture,
                              smoke_density, air_density, wind_speed,
                              ffmc, dmc, dc, isi, bui, fwi]])

    # Predict without button
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2}
    inverse_map = {v: k for k, v in risk_map.items()}

    xgb_pred_num = xgb_model.predict(input_values)[0]
    xgb_pred = inverse_map[xgb_pred_num]

    dt_pred_num = dt_model.predict(input_values)[0]
    dt_pred = inverse_map[dt_pred_num]

    st.subheader("🔥 Fire Risk Prediction (Auto-updated)")
    st.success(f"**XGBoost Prediction:** {xgb_pred}")
    st.success(f"**Decision Tree Prediction:** {dt_pred}")

    # Alert message
    if dt_pred == "High" or xgb_pred == "High":
        st.error("🚨 High Fire Risk Detected! Take Immediate Action!")
        # send_sms_alert("+919326735464", "High Fire Risk Detected! Take Immediate Action!")

    elif dt_pred == "Low" or xgb_pred == "Low":
        st.warning("⚠️ Low Fire Risk. Stay Alert.")
        # send_sms_alert("+919326735464", "Low Fire Risk. Stay Alert.")

    elif dt_pred == "Medium" or xgb_pred == "Medium":
        st.warning("⚠️ Moderate Fire Risk. Stay Alert.")
        # send_sms_alert("+919326735464", "Moderate Fire Risk. Stay Alert.")

    else:
        st.info("✅ Normal Conditions.")


else:
    st.error("❌ No sensor data found in Firebase.")