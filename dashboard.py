import streamlit as st
import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, db
from twilio.rest import Client
from streamlit_autorefresh import st_autorefresh
from collections import Counter
st.set_page_config(page_title="Forest Fire Dashboard", layout="wide")
firebase_config = st.secrets["firebase_service_account"]
st.write(firebase_config["project_id"])
   
# Models Load
xgb_model = joblib.load('fire_risk_xgb.pkl')
dt_model = joblib.load('decision_tree_model.pkl')
rf_model = joblib.load('random_forest_model.pkl')
cb_model = joblib.load('catboost_model.pkl')
lgbm_model = joblib.load('lightgbm_model.pkl')

# Auto-refresh every 5 seconds
st_autorefresh(interval=5000, key="auto_refresh")

# Fetch sensor data
sensor_ref = db.reference('/sensor')
sensor_data = sensor_ref.get()

# Initialize first dmc and dc and then fetching updated dmc and dc
previous_dmc = 25
previous_dc = 80
indices_ref = db.reference('/fire_indices')
indices_data = indices_ref.get() or {}

# Sensor Data
if sensor_data:    
    temp_c = sensor_data['tempBMP']
    humidity = sensor_data['humidity']
    pressure = sensor_data['pressure']
    soil_moisture = sensor_data['soil']
    smoke_density = sensor_data['mq7']
    mq5 = sensor_data['mq5']
    temp_k = temp_c + 273.15
    air_density = pressure * 100 / (287.05 * temp_k)
    wind_speed = 7.23

    # Fire Weather Index Calculations
    ffmc = (59.5 * (1 - (humidity / 100))) + (temp_c - 10) * 0.25 + wind_speed * 0.5
    k = (0.36 * (temp_c + 2)) * (1 - (humidity / 100.0)) * (12 / 30.0)
    dmc = previous_dmc + k
    dc = previous_dc + 0.36 * (temp_c + 2.8)
    isi = 0.45 * np.exp(0.05039 * ffmc) * (1 + (wind_speed ** 1.5 / 100)) 
    bui = dmc + (1 - (0.8 * dc) / (dmc + 0.4 * dc)) * (0.92 + (0.0114 * dmc) ** 1.7)  
    
    ab = 1000 / (25 + 108.64 * np.exp(-0.023 * bui))       
    B = 0.1 * isi * ab
    fwi = B
    ffmc, dmc, dc, isi, bui, fwi, k, temp_k, air_density = map(lambda x: round(max(x, 0), 2),[ffmc, dmc, dc, isi, bui, fwi, k, temp_k, air_density])
    
    # Save updated DMC and DC to Firebase
    indices_ref.update({
    'previous_dmc': dmc,
    'previous_dc': dc
    })

    # Input for model
    input_values = np.array([[temp_c, temp_k, humidity, pressure, soil_moisture,smoke_density, air_density, wind_speed,ffmc, dmc, dc, isi, bui, fwi]])

    # Model Prediction
    risk_map = {'Low': 0, 'Medium': 1, 'High': 2,'Extreme':3}
    inverse_map = {v: k for k, v in risk_map.items()}

    xgb_pred = inverse_map[xgb_model.predict(input_values)[0]]
    dt_pred = inverse_map[dt_model.predict(input_values)[0]]
    rf_pred = inverse_map[rf_model.predict(input_values)[0]]
    lgbm_pred = inverse_map[lgbm_model.predict(input_values)[0]]
    cb_pred = inverse_map[cb_model.predict(input_values).item()]
  

    dashboard_data = {       
        "Temp (°C)": temp_c,
        "Humidity (%)": humidity,
        "Pressure (hPa)": pressure,
        "Soil Moisture": soil_moisture,
        "Smoke Density (MQ-7)": smoke_density,
        "MQ-5":mq5,
        "Air Density": air_density,
        "Wind Speed (km/h)": wind_speed,
        "FFMC": ffmc,
        "DMC": dmc,
        "DC": dc,
        "ISI": isi,
        "BUI": bui,
        "FWI": fwi,        
        "XGBoost Prediction": xgb_pred,
        "Decision Tree Prediction": dt_pred,
        "Random Forest Prediction": rf_pred,
        "CatBoost Prediction":cb_pred,
        "LightGBM Prediction":lgbm_pred
    }

    labels = list(dashboard_data.keys())
    
    # First Row 
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("**Temp (°C)**")
        if temp_c < 35:
            bg_color = "#32CD32"  # Green
        elif 35 <= temp_c <= 45:
            bg_color = "#FFD700"  # Yellow
        else:
            bg_color = "#FF4500"  # Red
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{temp_c}</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Humidity %**")
        if humidity > 60:
            bg_color = "#32CD32"
        elif 30 <= humidity <= 60:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{humidity} </div>", unsafe_allow_html=True)

    with col3:
        st.markdown("**Pressure (hPa)**")
        if pressure >= 900:
            bg_color = "#32CD32"
        elif pressure < 500:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{pressure} </div>", unsafe_allow_html=True)

    with col4:
        st.markdown("**Temp (K)**")
        if temp_k < 308:
            bg_color = "#32CD32"
        elif 308 <= temp_k <= 318:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{temp_k} </div>", unsafe_allow_html=True)

    with col5:
        st.markdown("Soil Moisture ")
        if soil_moisture < 600:
            bg_color = "#32CD32"
        elif soil_moisture <= 800:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{soil_moisture} </div>", unsafe_allow_html=True)

  
    # Second Row
    col6, col7, col8, col9, col10 = st.columns(5)

    with col6:
        st.markdown("**Smoke Density (ppm)**")
        if smoke_density < 400:
            bg_color = "#32CD32"
        elif smoke_density <= 850:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{smoke_density} </div>", unsafe_allow_html=True)


    with col7:
        st.markdown("**MQ5(ppm)**")
        if mq5 < 500:
            bg_color = "#32CD32"
        elif mq5 <= 600:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{mq5} </div>", unsafe_allow_html=True)    

    with col8:
        st.markdown("**Air Density (kg/m³)**")
        if air_density > 1.05:
            bg_color = "#32CD32"
        elif air_density > 0.9:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(air_density, 3)} </div>", unsafe_allow_html=True)

    with col9:
        st.markdown("**Wind Speed (Km/h)**")
        if wind_speed < 8:
            bg_color = "#32CD32"
        elif  wind_speed <= 12:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(wind_speed, 2)} </div>", unsafe_allow_html=True)
     
    
    with col10:
        st.markdown("**FFMC**")
        if ffmc < 45:
            bg_color = "#32CD32"
        elif ffmc <= 75:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(ffmc, 2)}</div>", unsafe_allow_html=True)

    #Third Row
    col11, col12, col13, col14, col15 = st.columns(5)    

    with col11:
        st.markdown("**DMC**")
        if dmc < 35:
            bg_color = "#32CD32"
        elif  dmc <= 70:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(dmc, 2)}</div>", unsafe_allow_html=True)

    with col12:
        st.markdown("**DC**")
        if dc < 300:
            bg_color = "#32CD32"
        elif  dc <= 500:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(dc, 2)}</div>", unsafe_allow_html=True)


    with col13:
        st.markdown("**ISI**")
        if isi < 8:
            bg_color = "#32CD32"
        elif isi <= 16:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(isi, 2)}</div>", unsafe_allow_html=True)

    with col14:
        st.markdown("**BUI**")
        if bui < 50:
            bg_color = "#32CD32"
        elif  bui <= 90:
            bg_color = "#FFD700"
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(bui, 2)}</div>", unsafe_allow_html=True)

    with col15:
        st.markdown("**FWI**")
        if fwi < 12:
            bg_color = "#32CD32"
        elif fwi <= 27:
            bg_color = "#FFD700"
        elif fwi <= 36:
            bg_color = "#FF8C00"     
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{round(fwi, 2)}</div>", unsafe_allow_html=True)

    
    st.markdown("---")
    # Model Output
    col16, col17, col18, col19, col20 = st.columns(5)

    with col16:
        st.markdown("**XGBoost**")
        if xgb_pred == "Low":
            bg_color = "#32CD32"
        elif xgb_pred == "Medium":
            bg_color = "#FFD700"
        elif xgb_pred == "High":
            bg_color = "#FF8C00"    
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{xgb_pred}</div>", unsafe_allow_html=True)

    with col17:
        st.markdown("**Decision Tree**")
        if dt_pred == "Low":
            bg_color = "#32CD32"
        elif dt_pred == "Medium":
            bg_color = "#FFD700"
        elif dt_pred == "High":
            bg_color = "#FF8C00"    
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{dt_pred}</div>", unsafe_allow_html=True)


    with col18:
        st.markdown("**Random Forest**")
        if rf_pred == "Low":
            bg_color = "#32CD32"
        elif rf_pred == "Medium":
            bg_color = "#FFD700"
        elif rf_pred == "High":
            bg_color = "#FF8C00"    
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{rf_pred}</div>", unsafe_allow_html=True)

    with col19:
        st.markdown("**CatBoost**")
        if cb_pred == "Low":
            bg_color = "#32CD32"
        elif cb_pred == "Medium":
            bg_color = "#FFD700"
        elif cb_pred == "High":
            bg_color = "#FF8C00"    
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{cb_pred}</div>", unsafe_allow_html=True)

    with col20:
        st.markdown("**LightGBM**")
        if lgbm_pred == "Low":
            bg_color = "#32CD32"
        elif lgbm_pred == "Medium":
            bg_color = "#FFD700"
        elif lgbm_pred == "High":
            bg_color = "#FF8C00"    
        else:
            bg_color = "#FF4500"
        st.markdown(f"<div style='padding: 10px; background-color: {bg_color}; text-align: center; "f"border-radius: 8px; font-size: 18px; font-weight: bold;'>{lgbm_pred}</div>", unsafe_allow_html=True)    

    
    # Risk-based alert
    st.markdown("---")
    predictions = [xgb_pred, dt_pred, rf_pred, cb_pred,lgbm_pred]
    count = Counter(predictions)
    most_common_risk, freq = count.most_common(1)[0]

    if most_common_risk == "High":
        st.error("**High Fire Risk Detected!**")
    elif most_common_risk == "Medium":
        st.warning(" **Moderate Fire Risk.**")
    elif most_common_risk == "Extreme":
        st.error(" **Extreme Fire Risk.**")    
    else:
        st.success(" **Low Fire Risk. Conditions are stable.**")

    if mq5 > 100:
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
        send_sms_alert("+919326735464", "Fire Detected. Stay Safe.")

else:
    st.error(" No sensor data found in Firebase.")
