import pandas as pd
import random
import math

# Function to calculate Air Density
def calculate_air_density(pressure, temp_c):
    temp_k = temp_c + 273.15
    return round(pressure / temp_k, 2)

# Function to calculate Wind Speed (based on example formula)
def calculate_wind_speed(pressure1, pressure2, air_density):
    return round(math.sqrt((2 * abs(pressure2 - pressure1)) / air_density), 2)

# Function to calculate FFMC (Fine Fuel Moisture Code)
def calculate_ffmc(temp_c, humidity):
    return round(0.047 * temp_c + 0.054 * humidity + random.uniform(0, 10), 2)

# Function to calculate DMC (Duff Moisture Code)
def calculate_dmc(soil_moisture):
    return round(soil_moisture * random.uniform(0.5, 1.5), 2)

# Function to calculate DC (Drought Code)
def calculate_dc(soil_moisture):
    return round(soil_moisture * random.uniform(1.0, 2.0), 2)

# Function to calculate ISI (Initial Spread Index)
def calculate_isi(wind_speed, ffmc):
    return round(wind_speed * (ffmc / 10), 2)

# Function to calculate BUI (Buildup Index)
def calculate_bui(dmc, dc):
    return round((dmc + dc) / 2, 2)

# Function to calculate FWI (Fire Weather Index)
def calculate_fwi(isi, bui):
    return round(isi * (bui / 10), 2)

# Function to classify the fire risk based on FWI
def classify_risk(fwi):
    if fwi < 10:
        return "Low"
    elif 10 <= fwi <= 30:
        return "Medium"
    else:
        return "High"

# Generate synthetic dataset
data = []
for _ in range(8000):
    # Generate random values for temperature, humidity, pressure, soil moisture, and smoke density
    temp_c = round(random.uniform(10, 60), 2)  # Temperature in Celsius
    temp_k = round(temp_c + 273.15, 2)  # Temperature in Kelvin
    humidity = round(random.uniform(30, 90), 2)  # Humidity in percentage
    pressure = round(random.uniform(900, 1025), 2)  # Pressure in hPa
    soil_moisture = round(random.uniform(1, 40), 2)  # Soil moisture in percentage
    smoke_density = round(random.uniform(30, 70), 2)  # Smoke density in ppm

    # Calculate derived values
    air_density = calculate_air_density(pressure, temp_c)
    wind_speed = calculate_wind_speed(pressure, random.uniform(900, 1025), air_density)
    ffmc = calculate_ffmc(temp_c, humidity)
    dmc = calculate_dmc(soil_moisture)
    dc = calculate_dc(soil_moisture)
    isi = calculate_isi(wind_speed, ffmc)
    bui = calculate_bui(dmc, dc)
    fwi = calculate_fwi(isi, bui)
    risk = classify_risk(fwi)

    # Append data to the list
    data.append([temp_c, temp_k, humidity, pressure, soil_moisture, smoke_density, air_density, wind_speed, ffmc, dmc, dc, isi, bui, fwi, risk])

# Create DataFrame
df = pd.DataFrame(data, columns=["Temp(C)", "Temp(K)", "Humidity", "Pressure", "Soil Moisture", "Smoke Density", "Air Density", "Wind Speed", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI", "Risk"])

# Save to CSV
df.to_csv('fire_risk_dataset.csv', index=False)

# Display first few rows of the dataset
print(df.head())
