"""
Prediction of co2 values of scd30 sensor using sensor readings from bme680 sensor
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model, save_model
import os
from datetime import date, time, datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model
import io
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tempfile
from sklearn.cluster import KMeans
from scipy.stats import zscore

load_dotenv()

# Known table schema for sensor data
TABLES_SCHEMA = {
    'measurements': {
        'columns': ['id', 'node_id', 'timestamp'],
        'datetime_col': 'timestamp',
        'description': 'Main measurements table with timestamps'
    },
    'sensor_bme680': {
        'columns': ['id', 'measurement_id', 'humidity', 'temperature', 'gas_resistance', 'pressure'],
        'join_col': 'measurement_id',
        'description': 'BME680 sensor data (humidity, temperature, gas, pressure)',
        'metrics': ['humidity', 'temperature', 'gas_resistance', 'pressure']
    },
    'sensor_scd30': {
        'columns': ['id', 'measurement_id', 'co2', 'temperature', 'humidity'],
        'join_col': 'measurement_id',
        'description': 'SCD30 sensor data (CO2, temperature, humidity)',
        'metrics': ['co2', 'temperature', 'humidity']
    },
    'sensor_sps30': {
        'columns': ['id', 'measurement_id', 'mass_conc_pm1', 'mass_conc_pm2_5', 'mass_conc_pm4', 'mass_conc_pm10', 'num_conc_pm0_5', 'num_conc_pm1', 'num_conc_pm2_5', 'num_conc_pm4', 'num_conc_pm10', 'particle_size'],
        'join_col': 'measurement_id',
        'description': 'SPS30 sensor data (particulate matter concentrations)',
        'metrics': ['mass_conc_pm2_5', 'mass_conc_pm10', 'particle_size']
    }
}


def filter_by_date(data, start_date, end_date):
    return data[(data["timestamp"] >= start_date) & (data["timestamp"] <= end_date)]



def detect_process_days(df):
    df = df.copy()
    df = df.dropna()
    
    # Extract time features
    df["hour"] = df["timestamp"].dt.hour
    df["working_hours"] = df["hour"].between(8, 18)

    # Columns to normalize
    features_to_scale = [
        'gas_resistance', 'co2',
        'mass_conc_pm2_5', 'mass_conc_pm10',
        'num_conc_pm2_5', 'num_conc_pm10',
        'particle_size'
    ]
    
    # Time columns to keep
    time_columns = ['timestamp', 'hour', 'working_hours']

    # Scale features
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[features_to_scale])
    df_scaled_features = pd.DataFrame(scaled_values, columns=features_to_scale, index=df.index)

    # Filter out outliers (z-score < 3)
    z_scores = df_scaled_features.apply(zscore)
    keep_rows = (z_scores.abs() < 3).all(axis=1)
    
    df_scaled = pd.concat([df[time_columns], df_scaled_features], axis=1)
    df_no_outliers = df_scaled[keep_rows]
    df_no_outliers["date"] = df_no_outliers["timestamp"].dt.date

    # Enhanced daily summary
    def summarize_day(group):
        working = group[group["working_hours"]]
        non_working = group[~group["working_hours"]]

        summary = {}
        for feature in features_to_scale:
            summary[f"{feature}_working_mean"] = working[feature].mean()
            summary[f"{feature}_non_working_mean"] = non_working[feature].mean()
            summary[f"{feature}_working_std"] = working[feature].std()
            summary[f"{feature}_non_working_std"] = non_working[feature].std()
        return pd.Series(summary)
    

    enhanced_daily = df_no_outliers.groupby("date").apply(summarize_day)
    daily_mean = df_no_outliers.groupby("date")[features_to_scale].mean()

    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    numeric_cols = enhanced_daily.select_dtypes(include=['number']).columns
    numeric_cols = enhanced_daily.select_dtypes(include=['number']).columns
    enhanced_daily[numeric_cols] = enhanced_daily[numeric_cols].fillna(enhanced_daily[numeric_cols].mean())
    clusters = kmeans.fit_predict(enhanced_daily[numeric_cols])

    enhanced_daily["cluster"] = clusters
    daily_mean["cluster"] = clusters

    # Label clusters
    cluster_summary = enhanced_daily.groupby("cluster").mean()
    if cluster_summary["co2_working_mean"].iloc[0] < cluster_summary["co2_working_mean"].iloc[1]:
        cluster_mapping = {0: "baseline", 1: "process"}
    else:
        cluster_mapping = {0: "process", 1: "baseline"}

    enhanced_daily["label"] = enhanced_daily["cluster"].map(cluster_mapping)
    daily_mean["label"] = daily_mean["cluster"].map(cluster_mapping)

    # Extract dates
    process_days = enhanced_daily[enhanced_daily["label"] == "process"].index
    baseline_days = enhanced_daily[enhanced_daily["label"] == "baseline"].index

    return process_days



def cyclic_encode(series, max_val):
    return np.sin(2 * np.pi * series / max_val), np.cos(2 * np.pi * series / max_val)


def get_date_range_for_table(conn, default_days=30):
    """Get appropriate date range for measurements table"""
    try:
        # Get data range from measurements table
        date_query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM measurements"
        date_df = pd.read_sql_query(date_query, conn)
        
        if not date_df.empty and date_df['min_date'].iloc[0]:
            min_date = pd.to_datetime(date_df['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(date_df['max_date'].iloc[0]).date()
            
            # Default to last month of available data
            default_start = max_date - timedelta(days=default_days)
            if default_start < min_date:
                default_start = min_date
                
            return min_date, max_date, default_start, max_date
        else:
            # Fallback for empty data - use 2021-2022 range
            return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date(), datetime(2022, 12, 1).date(), datetime(2022, 12, 31).date()
    except:
        # Fallback dates for 2021-2022
        return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date(), datetime(2022, 12, 1).date(), datetime(2022, 12, 31).date()


def cyclic_encode(value, max_val):
    sin_val = np.sin(2 * np.pi * value / max_val)
    cos_val = np.cos(2 * np.pi * value / max_val)
    return sin_val, cos_val

def detect_process_days(df):
    df = df.copy()
    df = df.dropna()
    
    # Extract time features
    df["hour"] = df["timestamp"].dt.hour
    df["working_hours"] = df["hour"].between(8, 18)

    # Columns to normalize
    features_to_scale = [
        'gas_resistance', 'co2',
        'mass_conc_pm2_5', 'mass_conc_pm10',
        'num_conc_pm2_5', 'num_conc_pm10',
        'particle_size'
    ]
    
    # Time columns to keep
    time_columns = ['timestamp', 'hour', 'working_hours']

    # Scale features
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[features_to_scale])
    df_scaled_features = pd.DataFrame(scaled_values, columns=features_to_scale, index=df.index)

    # Filter out outliers (z-score < 3)
    z_scores = df_scaled_features.apply(zscore)
    keep_rows = (z_scores.abs() < 3).all(axis=1)
    
    df_scaled = pd.concat([df[time_columns], df_scaled_features], axis=1)
    df_no_outliers = df_scaled[keep_rows]
    df_no_outliers["date"] = df_no_outliers["timestamp"].dt.date

    # Enhanced daily summary
    def summarize_day(group):
        working = group[group["working_hours"]]
        non_working = group[~group["working_hours"]]

        summary = {}
        for feature in features_to_scale:
            summary[f"{feature}_working_mean"] = working[feature].mean()
            summary[f"{feature}_non_working_mean"] = non_working[feature].mean()
            summary[f"{feature}_working_std"] = working[feature].std()
            summary[f"{feature}_non_working_std"] = non_working[feature].std()
        return pd.Series(summary)
    

    enhanced_daily = df_no_outliers.groupby("date").apply(summarize_day)
    daily_mean = df_no_outliers.groupby("date")[features_to_scale].mean()

    # Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    numeric_cols = enhanced_daily.select_dtypes(include=['number']).columns
    numeric_cols = enhanced_daily.select_dtypes(include=['number']).columns
    enhanced_daily[numeric_cols] = enhanced_daily[numeric_cols].fillna(enhanced_daily[numeric_cols].mean())
    clusters = kmeans.fit_predict(enhanced_daily[numeric_cols])

    enhanced_daily["cluster"] = clusters
    daily_mean["cluster"] = clusters

    # Label clusters
    cluster_summary = enhanced_daily.groupby("cluster").mean()
    if cluster_summary["co2_working_mean"].iloc[0] < cluster_summary["co2_working_mean"].iloc[1]:
        cluster_mapping = {0: "baseline", 1: "process"}
    else:
        cluster_mapping = {0: "process", 1: "baseline"}

    enhanced_daily["label"] = enhanced_daily["cluster"].map(cluster_mapping)
    daily_mean["label"] = daily_mean["cluster"].map(cluster_mapping)

    # Extract dates
    process_days = enhanced_daily[enhanced_daily["label"] == "process"].index
    baseline_days = enhanced_daily[enhanced_daily["label"] == "baseline"].index

    return process_days


def show():
    st.header("📈 CO₂ Prediction using BME680 Readings")
    st.markdown("Predict CO₂ levels from the SCD30 sensor based on features from the BME680 sensor.")
    
    db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')

    st.subheader("🧠 Model Overview")
    st.markdown("""
    We have trained two neural network models to predict **CO₂ levels** from **BME680 sensor data** for the IMF lab data:""")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🤖 Model A")
        st.markdown("""
        **Features:**  
        - `gas_resistance`  
        - Time features (`month`, `week number`, `day of week`, `hour`, etc.)

        **Accuracy:**  
        - ✅ 73.05% within ±5% tolerance  
        - ✅ 94.17% within ±10% tolerance
        """)

    with col2:
        st.markdown("### 🧪 Model B")
        st.markdown("""
        **Features:**  
        - `gas_resistance`, `temperature`, `humidity`, `pressure`  
        - Time features (`month`, `week number`, `day of week`, `hour`, etc.)

        **Accuracy:**  
        - ✅ 76.19% within ±5% tolerance  
        - ✅ 95.91% within ±10% tolerance
        """)

    st.markdown("You can either select one of the pre-trained models to predict CO₂ levels for the IMF Lab, or if you are working with data from a different setting, you can train a custom model based on your selected data.")

    #model_choice = st.radio(
    #    "🔧 What would you like to do?",
    #    ["Use Pre-trained Model", "Train New Model on Selected Data"],
    #    index=0,
    #    horizontal=True
    #)

    mode = st.selectbox(
    "Select Mode",
    ["-- Select Mode --", "Use Pre-trained Model", "Train New Model on Selected Data", "Use the model you trained"]
    )

    if mode == "Use Pre-trained Model":
        st.subheader("📦 Choose a Pre-trained Model")

        model_type = st.selectbox(
        "Select Model",
        ["-- Select Model --", "Model A (Gas Resistance + Time)", "Model B (All BME680 Features + Time)"]
        )

        if model_type == "Model A (Gas Resistance + Time)":
            st.info("🔄 Loading Model A...")
            try:
                with st.spinner("Loading Model A files..."):
                    #st.write("Current working directory:", os.getcwd())

                    model = load_model('pretrained/all_data_model_just_gas.h5')  # Keras model
                    scaler = joblib.load('pretrained/all_data_scaler_just_gas.pkl')  # Feature scaler
                st.success("Model A loaded successfully!")
            
            except Exception as e:
                st.error(f"Failed to load Model A: {e}")
            
            
            try:
                prediction_type = st.selectbox(
                    "Select How You Would Like To Test the Model",
                    ["-- Select Option --", "Entering Details for a Single Row", "Uploading a CSV file", "Selecting a Day from the Database"]
                )

                if prediction_type == "Entering Details for a Single Row":            
                                
                    st.subheader("📝 Provide Input for Prediction")



                    # Input fields
                    node_id = st.selectbox("Select Node ID", options=[6,7,8,9,10,11,12], index=0)
                    st.subheader("📅 Enter Timestamp and Node Info")

                    # Ask for date and time separately
                    input_date = st.date_input("Select Date", value=date.today())
                    hour_text = st.text_input("Enter Hour (0–23)")
                    try:
                        hour = int(hour_text)
                        if not 0 <= hour <= 23:
                            st.error("Hour must be between 0 and 23.")
                            hour = None
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None
                
                    gas_resistance_text = st.text_input("Enter gas resistance.")
                    try:
                        gas_resistance = float(gas_resistance_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    if st.button("📈 Predict CO₂ Level"):
                        if gas_resistance is None or input_date is None or node_id is None or hour is None:
                            st.warning("Please enter all fields.")
                        else:
                            try:
                                # Extract time features
                                week_number = input_date.isocalendar().week
                                hour = hour
                                dayofweek = input_date.weekday()
                                month = input_date.month

                                # Cyclic encodings
                                hour_sin, hour_cos = cyclic_encode(hour, 24)
                                dayofweek_sin, dayofweek_cos = cyclic_encode(dayofweek, 7)
                                month_sin, month_cos = cyclic_encode(month, 12)
                                week_sin, week_cos = cyclic_encode(week_number, 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in range(6, 13)]
                                node_encoding = {col: False for col in node_cols}
                                if f"node_{node_id}" in node_encoding:
                                    node_encoding[f"node_{node_id}"] = True

                                # Create final DataFrame row
                                input_dict = {
                                    "gas_resistance": gas_resistance,
                                    "week_number": week_number,
                                    "hour_sin": hour_sin,
                                    "hour_cos": hour_cos,
                                    "dayofweek_sin": dayofweek_sin,
                                    "dayofweek_cos": dayofweek_cos,
                                    "month_sin": month_sin,
                                    "month_cos": month_cos,
                                    "week_number_sin": week_sin,
                                    "week_number_cos": week_cos,
                                    **node_encoding
                                }

                                input_df = pd.DataFrame([input_dict])

                                # ⚠️ Ensure feature columns match training order (optional but safer)
                                expected_columns = [
                                    'gas_resistance', 'week_number', 'hour_sin', 'hour_cos',
                                    'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
                                    'week_number_sin', 'week_number_cos',
                                    'node_6', 'node_7', 'node_8', 'node_9', 'node_10', 'node_11', 'node_12'
                                ]
                                input_df = input_df[expected_columns]

                                # Scale input
                                scaled_input = scaler.transform(input_df)

                                # Predict
                                predicted_co2 = model.predict(scaled_input)[0][0]

                                st.success(f"🌿 Predicted CO₂ Level: **{predicted_co2:.2f} ppm**")
                        
                            except Exception as e:
                                st.error(f"❌ Sinlge Row Prediction failed: {e}")
                
                elif prediction_type == "Uploading a CSV file":
                    st.subheader("📁 Upload CSV for Batch Prediction")
                    uploaded_file = st.file_uploader("Upload a CSV file with a 'timestamp', 'gas_resistance', and 'node_id' column", type=["csv"])
    
                    if uploaded_file:
                        try:
                            df = pd.read_csv(uploaded_file)

                            # Validate required columns
                            required_cols = {'timestamp', 'gas_resistance', 'node_id'}
                            if not required_cols.issubset(df.columns):
                                st.error(f"CSV must contain columns: {required_cols}")
                            else:
                                # Convert timestamp column to datetime
                                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                                df.dropna(subset=["timestamp", "gas_resistance", "node_id"], inplace=True)

                                # Extract time features
                                df["hour"] = df["timestamp"].dt.hour
                                df["dayofweek"] = df["timestamp"].dt.weekday
                                df["month"] = df["timestamp"].dt.month
                                df["week_number"] = df["timestamp"].dt.isocalendar().week

                                # Cyclic encodings
                                df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour"], 24)
                                df["dayofweek_sin"], df["dayofweek_cos"] = cyclic_encode(df["dayofweek"], 7)
                                df["month_sin"], df["month_cos"] = cyclic_encode(df["month"], 12)
                                df["week_number_sin"], df["week_number_cos"] = cyclic_encode(df["week_number"], 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in range(6, 13)]
                                for col in node_cols:
                                    df[col] = (df["node_id"] == int(col.split("_")[1])).astype(bool)

                                # Final feature list
                                expected_columns = [
                                    'gas_resistance', 'week_number',
                                    'hour_sin', 'hour_cos',
                                    'dayofweek_sin', 'dayofweek_cos',
                                    'month_sin', 'month_cos',
                                    'week_number_sin', 'week_number_cos',
                                    *node_cols
                                ]

                                # Scale and predict
                                X = df[expected_columns]
                                X_scaled = scaler.transform(X)
                                predictions = model.predict(X_scaled).flatten()
                                df["predicted_co2"] = predictions

                                st.success("✅ Predictions completed!")
                                st.dataframe(df.head())

                                st.download_button(
                                    label="📥 Download CSV with Predictions",
                                    data=df.to_csv(index=False),
                                    file_name="predicted_co2.csv",
                                    mime="text/csv"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Failed to process file: {e}")

                elif prediction_type == "Selecting a Day from the Database":
                    st.subheader("📊 Compare Predicted vs Actual CO₂")
    
                    # User selects
                    
                    if not os.path.exists(db_path):
                        st.error("Database file not found. Please check your .env configuration.")
                        return
        
                    try:
                        conn = sqlite3.connect(db_path)
        
                        # Get available date range
                        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)

                        selected_date = st.date_input(
                            "📅 Select Date",
                            value=default_start,
                            min_value=min_date,
                            max_value=max_date,
                            key="db_start_date"
                        )
                        selected_date = pd.to_datetime(selected_date)
                        start_dt = selected_date.strftime("%Y-%m-%d") + " 00:00:00"
                        end_dt = selected_date.strftime("%Y-%m-%d") + " 23:59:59"
                        selected_node = st.selectbox("Select Node ID", options=range(6, 13))
        


                    
                        query = f"""
                            SELECT m.timestamp, m.node_id, b.gas_resistance, s.co2
                            FROM measurements m
                            JOIN sensor_bme680 b ON m.id = b.measurement_id
                            JOIN sensor_scd30 s ON m.id = s.measurement_id
                            WHERE m.timestamp >= '{start_dt}'
                            AND m.timestamp <= '{end_dt}'
                            AND m.node_id = {selected_node}
                            ORDER BY m.timestamp
                            """
                        df_day = pd.read_sql_query(query,conn)
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day = df_day[(df_day["timestamp"].dt.hour >= 9) & (df_day["timestamp"].dt.hour < 18)]

                        if df_day.empty:
                                st.warning("No data found for the selected date and node.")
                                return
                        cols_to_scale = ['co2']
                        cols_to_scale_present = [col for col in cols_to_scale if col in df_day.columns]
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present] * 1e-2

                        # Feature engineering
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day["hour"] = df_day["timestamp"].dt.hour
                        df_day["dayofweek"] = df_day["timestamp"].dt.weekday
                        df_day["month"] = df_day["timestamp"].dt.month
                        df_day["week_number"] = df_day["timestamp"].dt.isocalendar().week

                        # Cyclic encodings
                        df_day["hour_sin"], df_day["hour_cos"] = cyclic_encode(df_day["hour"], 24)
                        df_day["dayofweek_sin"], df_day["dayofweek_cos"] = cyclic_encode(df_day["dayofweek"], 7)
                        df_day["month_sin"], df_day["month_cos"] = cyclic_encode(df_day["month"], 12)
                        df_day["week_number_sin"], df_day["week_number_cos"] = cyclic_encode(df_day["week_number"], 52)

                        # One-hot encode node
                        for i in range(6, 13):
                            df_day[f"node_{i}"] = (df_day["node_id"] == i).astype(bool)

                        # Prepare feature matrix
                        expected_columns = [
                            'gas_resistance', 'week_number',
                            'hour_sin', 'hour_cos',
                            'dayofweek_sin', 'dayofweek_cos',
                            'month_sin', 'month_cos',
                            'week_number_sin', 'week_number_cos',
                            'node_6', 'node_7', 'node_8', 'node_9', 'node_10', 'node_11', 'node_12'
                        ]
                        X = df_day[expected_columns]
                        X_scaled = scaler.transform(X)

                        # Predict
                        df_day["predicted_co2"] = model.predict(X_scaled).flatten()

                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df_day["timestamp"], df_day["co2"], label="Actual CO₂", marker='o', linestyle='-')
                        ax.plot(df_day["timestamp"], df_day["predicted_co2"], label="Predicted CO₂", marker='x', linestyle='--')
                        ax.set_title(f"CO₂: Prediction vs Actual | Node {selected_node} | {selected_date.date()}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("CO₂ (ppm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"❌ Failed to generate plot: {e}")
                    finally:
                        if 'conn' in locals():
                            conn.close()

            except Exception as e:
                st.error(f"❌ Testing failed: {e}")

        elif model_type == "Model B (All BME680 Features + Time)":
            st.info("🔄 Loading Model B...")
            try:
                with st.spinner("Loading Model B files..."):
                    model = load_model('pretrained/all_data_all_features_model.h5')
                    scaler = joblib.load('pretrained/all_data_all_features_scaler.pkl')
                st.success("Model B loaded successfully!")
            
            except Exception as e:
                st.error(f"Failed to load Model B: {e}")

            try:
                prediction_type = st.selectbox(
                    "Select How You Would Like To Test the Model",
                    ["-- Select Option --", "Entering Details for a Single Row", "Uploading a CSV file", "Selecting a Day from the Database"]
                )

                if prediction_type == "Entering Details for a Single Row":

                    st.subheader("📝 Provide Input for Prediction")

                    # Input fields
                    node_id = st.selectbox("Select Node ID", options=[6,7,8,9,10,11,12], index=0)
                    st.subheader("📅 Enter Timestamp and Node Info")

                    # Ask for date and time separately
                    input_date = st.date_input("Select Date", value=date.today())
                    hour_text = st.text_input("Enter Hour (0–23)")
                    try:
                        hour = int(hour_text)
                        if not 0 <= hour <= 23:
                            st.error("Hour must be between 0 and 23.")
                            hour = None
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None
                
                    gas_resistance_text = st.text_input("Enter gas resistance.")
                    try:
                        gas_resistance = float(gas_resistance_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    temperature_text = st.text_input("Enter temperature.")
                    try:
                        temperature_bme = float(temperature_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    humidity_text = st.text_input("Enter humidity.")
                    try:
                        humidity_bme = float(humidity_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    pressure_text = st.text_input("Enter pressure.")
                    try:
                        pressure = float(pressure_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None



                    if st.button("📈 Predict CO₂ Level"):
                        if temperature_bme is None or humidity_bme is None or pressure is None or gas_resistance is None or input_date is None or node_id is None or hour is None:
                            st.warning("Please enter all fields.")
                        else:
                            try:
                                # Extract time features
                                week_number = input_date.isocalendar().week
                                hour = hour
                                dayofweek = input_date.weekday()
                                month = input_date.month

                                # Cyclic encodings
                                hour_sin, hour_cos = cyclic_encode(hour, 24)
                                dayofweek_sin, dayofweek_cos = cyclic_encode(dayofweek, 7)
                                month_sin, month_cos = cyclic_encode(month, 12)
                                week_sin, week_cos = cyclic_encode(week_number, 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in range(6, 13)]
                                node_encoding = {col: False for col in node_cols}
                                if f"node_{node_id}" in node_encoding:
                                    node_encoding[f"node_{node_id}"] = True

                                # Create final DataFrame row
                                input_dict = {
                                    "temperature_bme": temperature_bme,
                                    "humidity_bme": humidity_bme,
                                    "pressure": pressure,
                                    "gas_resistance": gas_resistance,
                                    "week_number": week_number,
                                    "hour_sin": hour_sin,
                                    "hour_cos": hour_cos,
                                    "dayofweek_sin": dayofweek_sin,
                                    "dayofweek_cos": dayofweek_cos,
                                    "month_sin": month_sin,
                                    "month_cos": month_cos,
                                    "week_number_sin": week_sin,
                                    "week_number_cos": week_cos,
                                    **node_encoding
                                }

                                input_df = pd.DataFrame([input_dict])

                                # ⚠️ Ensure feature columns match training order (optional but safer)
                                expected_columns = [
                                    'temperature_bme', 'humidity_bme', 'pressure', 'gas_resistance', 'week_number', 'hour_sin', 'hour_cos',
                                    'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos',
                                    'week_number_sin', 'week_number_cos',
                                    'node_6', 'node_7', 'node_8', 'node_9', 'node_10', 'node_11', 'node_12'
                                ]
                                input_df = input_df[expected_columns]

                                # Scale input
                                scaled_input = scaler.transform(input_df)

                                # Predict
                                predicted_co2 = model.predict(scaled_input)[0][0]

                                st.success(f"🌿 Predicted CO₂ Level: **{predicted_co2:.2f} ppm**")
                        
                            except Exception as e:
                                st.error(f"❌ Single Row Prediction failed: {e}")

                elif  prediction_type == "Uploading a CSV file":
                    st.subheader("📁 Upload CSV for Batch Prediction")
                    uploaded_file = st.file_uploader("Upload a CSV file with a 'timestamp', 'temperature_bme', 'humidity_bme', 'pressure', 'gas_resistance' and 'node_id' column", type=["csv"])
    
                    if uploaded_file:
                        try:
                            df = pd.read_csv(uploaded_file)

                            # Validate required columns
                            required_cols = {'timestamp', 'temperature_bme', 'humidity_bme', 'pressure', 'gas_resistance', 'node_id'}
                            if not required_cols.issubset(df.columns):
                                st.error(f"CSV must contain columns: {required_cols}")
                            else:
                                # Convert timestamp column to datetime
                                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                                df.dropna(subset=["timestamp", "gas_resistance", "temperature_bme", "humidity_bme", "pressure", "node_id"], inplace=True)

                                # Extract time features
                                df["hour"] = df["timestamp"].dt.hour
                                df["dayofweek"] = df["timestamp"].dt.weekday
                                df["month"] = df["timestamp"].dt.month
                                df["week_number"] = df["timestamp"].dt.isocalendar().week

                                # Cyclic encodings
                                df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour"], 24)
                                df["dayofweek_sin"], df["dayofweek_cos"] = cyclic_encode(df["dayofweek"], 7)
                                df["month_sin"], df["month_cos"] = cyclic_encode(df["month"], 12)
                                df["week_number_sin"], df["week_number_cos"] = cyclic_encode(df["week_number"], 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in range(6, 13)]
                                for col in node_cols:
                                    df[col] = (df["node_id"] == int(col.split("_")[1])).astype(bool)

                                # Final feature list
                                expected_columns = [
                                    'temperature_bme', 'humidity_bme', 'pressure', 'gas_resistance', 'week_number',
                                    'hour_sin', 'hour_cos',
                                    'dayofweek_sin', 'dayofweek_cos',
                                    'month_sin', 'month_cos',
                                    'week_number_sin', 'week_number_cos',
                                    *node_cols
                                ]

                                # Scale and predict
                                X = df[expected_columns]
                                X_scaled = scaler.transform(X)
                                predictions = model.predict(X_scaled).flatten()
                                df["predicted_co2"] = predictions

                                st.success("✅ Predictions completed!")
                                st.dataframe(df.head())

                                st.download_button(
                                    label="📥 Download CSV with Predictions",
                                    data=df.to_csv(index=False),
                                    file_name="predicted_co2.csv",
                                    mime="text/csv"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Failed to process file: {e}")

                elif prediction_type == "Selecting a Day from the Database":
                    st.subheader("📊 Compare Predicted vs Actual CO₂")
    
                    # User selects
                    
                    if not os.path.exists(db_path):
                        st.error("Database file not found. Please check your .env configuration.")
                        return
        
                    try:
                        conn = sqlite3.connect(db_path)
        
                        # Get available date range
                        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)

                        selected_date = st.date_input(
                            "📅 Select Date",
                            value=default_start,
                            min_value=min_date,
                            max_value=max_date,
                            key="db_start_date"
                        )
                        selected_date = pd.to_datetime(selected_date)
                        start_dt = selected_date.strftime("%Y-%m-%d") + " 00:00:00"
                        end_dt = selected_date.strftime("%Y-%m-%d") + " 23:59:59"
                        selected_node = st.selectbox("Select Node ID", options=range(6, 13))
        


                    
                        query = f"""
                            SELECT m.timestamp, m.node_id, b.temperature as temperature_bme, b.humidity as humidity_bme, b.pressure, b.gas_resistance, s.co2
                            FROM measurements m
                            JOIN sensor_bme680 b ON m.id = b.measurement_id
                            JOIN sensor_scd30 s ON m.id = s.measurement_id
                            WHERE m.timestamp >= '{start_dt}'
                            AND m.timestamp <= '{end_dt}'
                            AND m.node_id = {selected_node}
                            ORDER BY m.timestamp
                            """
                        df_day = pd.read_sql_query(query,conn)
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day = df_day[(df_day["timestamp"].dt.hour >= 9) & (df_day["timestamp"].dt.hour < 18)]

                        if df_day.empty:
                                st.warning("No data found for the selected date and node.")
                                return
                        cols_to_scale = ['temperature_bme', 'humidity_bme', 'pressure', 'co2']
                        cols_to_scale_present = [col for col in cols_to_scale if col in df_day.columns]
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present] * 1e-2

                        # Feature engineering
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day["hour"] = df_day["timestamp"].dt.hour
                        df_day["dayofweek"] = df_day["timestamp"].dt.weekday
                        df_day["month"] = df_day["timestamp"].dt.month
                        df_day["week_number"] = df_day["timestamp"].dt.isocalendar().week

                        # Cyclic encodings
                        df_day["hour_sin"], df_day["hour_cos"] = cyclic_encode(df_day["hour"], 24)
                        df_day["dayofweek_sin"], df_day["dayofweek_cos"] = cyclic_encode(df_day["dayofweek"], 7)
                        df_day["month_sin"], df_day["month_cos"] = cyclic_encode(df_day["month"], 12)
                        df_day["week_number_sin"], df_day["week_number_cos"] = cyclic_encode(df_day["week_number"], 52)

                        # One-hot encode node
                        for i in range(6, 13):
                            df_day[f"node_{i}"] = (df_day["node_id"] == i).astype(bool)

                        # Prepare feature matrix
                        expected_columns = [
                            'temperature_bme', 'humidity_bme', 'pressure', 'gas_resistance', 'week_number',
                            'hour_sin', 'hour_cos',
                            'dayofweek_sin', 'dayofweek_cos',
                            'month_sin', 'month_cos',
                            'week_number_sin', 'week_number_cos',
                            'node_6', 'node_7', 'node_8', 'node_9', 'node_10', 'node_11', 'node_12'
                        ]
                        X = df_day[expected_columns]
                        X_scaled = scaler.transform(X)

                        # Predict
                        df_day["predicted_co2"] = model.predict(X_scaled).flatten()

                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df_day["timestamp"], df_day["co2"], label="Actual CO₂", marker='o', linestyle='-')
                        ax.plot(df_day["timestamp"], df_day["predicted_co2"], label="Predicted CO₂", marker='x', linestyle='--')
                        ax.set_title(f"CO₂: Prediction vs Actual | Node {selected_node} | {selected_date.date()}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("CO₂ (ppm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"❌ Failed to generate plot: {e}")
                    finally:
                        if 'conn' in locals():
                            conn.close()

            except Exception as e:
                st.error(f"❌ Testing failed: {e}")
                

        elif model_type == "-- Select Model --":
            st.info("Please select a model to continue.")

    elif mode == "Train New Model on Selected Data":
        st.subheader("🧠 Train New Model")
    
       
        if not os.path.exists(db_path):
            st.error("Database file not found. Please check your .env configuration.")
            return
        
        try:
            conn = sqlite3.connect(db_path)
        
            # Get available tables
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            available_tables = [table[0] for table in cursor.fetchall()]
        
            if not available_tables:
                st.warning("No tables found in the database.")
                conn.close()
                return
                    
            st.subheader("📅 Select date range of data you want to use to train the model")
        
            min_date, max_date, default_start, default_end = get_date_range_for_table(conn)
        
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date:",
                    value=default_start,
                    min_value=min_date,
                    max_value=max_date,
                    key="db_start_date"
                )
                start_date = pd.to_datetime(start_date)
        
            with col2:
                end_date = st.date_input(
                    "End Date:",
                    value=default_end,
                    min_value=min_date,
                    max_value=max_date,
                    key="db_end_date"
                )
                end_date = pd.to_datetime(end_date)
        
            # Validate date range
            if start_date > end_date:
                st.error("Start date must be before end date.")
                conn.close()
                
            query = f"""SELECT DISTINCT node_id AS unique_node_id FROM measurements"""
            available_node_ids = pd.read_sql_query(query, conn)
            available_node_ids = sorted(available_node_ids['unique_node_id'])


            
            
            with st.expander("✅ Node Selection", expanded=True):
                # Hardcoded default nodes
                default_nodes = [9, 12, 10, 6, 8, 11, 7]

                # Filter defaults to ensure they exist in the data
                valid_default_nodes = [node for node in default_nodes if node in available_node_ids]

                # Sidebar multiselect
                nodes = st.multiselect(
                    "Select node(s) to include",
                    options=available_node_ids,
                    default=valid_default_nodes
                )
                
                if not nodes:
                    st.warning("Please select at least one node to continue.")

                nodes = sorted(nodes)
                    
            
            



            st.subheader("📦 Choose how you want to train your model.")

            model_train_type = st.selectbox(
                "Select Model type",
                ["-- Select Model type --", "Model A (Gas Resistance + Time)", "Model B (All BME680 Features + Time)"]
                )
            
            if model_train_type == "-- Select Model type --":
                st.info("Please select a model_type to continue.")

            num_of_epochs = st.text_input("Enter Number of Epochs you want to train the model for:")
            try:
                num_of_epochs = int(num_of_epochs)
            except ValueError:
                st.error("Please enter a valid number.")
                num_of_epochs = None

            if st.button("Create model"):                           


                try:
                    with st.spinner("Getting Process dates..."):
                        query = f"""SELECT * from measurements m where timestamp >= '{start_date}' AND m.timestamp <= '{end_date} 23:59:59'"""
                        data = pd.read_sql_query(query, conn)
                        data["timestamp"] = pd.to_datetime(data["timestamp"])
                        st.write("Reading measurements table complete✅")


                        query = f"""SELECT * from sensor_bme680 where measurement_id IN (SELECT DISTINCT id from measurements m where timestamp >= '{start_date}' AND m.timestamp <= '{end_date} 23:59:59')"""
                        bme680 = pd.read_sql_query(query, conn)
                        bme680 = bme680.rename(columns={"id": "idbme", "measurement_id": "id"})
                        data = data.merge(bme680, on="id", how="left")
                        st.write("Reading sensor_bme680 table complete ✅")

                        query = f"""SELECT * from sensor_scd30 where measurement_id IN (SELECT DISTINCT id from measurements m where timestamp >= '{start_date}' AND m.timestamp <= '{end_date} 23:59:59')"""
                        scd30 = pd.read_sql_query(query, conn)
                        scd30 = scd30.rename(columns={"id": "idscd", "measurement_id": "id", "temperature": "temperature_scd", "humidity": "humidity_scd"})
                        data = data.merge(scd30, on="id", how="left")
                        st.write("Reading sensor_scd30 table complete ✅")

                        query = f"""SELECT * from sensor_sps30 where measurement_id IN (SELECT DISTINCT id from measurements m where timestamp >= '{start_date}' AND m.timestamp <= '{end_date} 23:59:59')"""
                        sps30 = pd.read_sql_query(query, conn)
                        sps30 = sps30.rename(columns={"id": "idsps", "measurement_id": "id"})
                        data = data.merge(sps30, on="id", how="left")
                        st.write("Reading sensor_sps30 table complete ✅")

                        data.drop(columns=['idbme', 'idsps', 'idscd', 'num_conc_pm4', 'num_conc_pm0_5', 'num_conc_pm1', 'mass_conc_pm4', 'mass_conc_pm1'], inplace=True)

                        cols_to_scale = ['co2', 'temperature', 'humidity', 'temperature_scd', 'humidity_scd', 'pressure', 'mass_conc_pm2_5', 'mass_conc_pm10', 'num_conc_pm2_5', 'num_conc_pm10']
                        
                        # Ensure only existing and numeric columns are scaled
                        cols_to_scale_present = [col for col in cols_to_scale if col in data.columns]
                        data[cols_to_scale_present] = data[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                        data[cols_to_scale_present] = data[cols_to_scale_present] * 1e-2


                        chunk_start = start_date
                        
                        list_process_days = []
                        while chunk_start < end_date:
                            if chunk_start.month == 12:
                                chunk_end = datetime(chunk_start.year + 1, 1, 1)
                            else:
                                chunk_end = datetime(chunk_start.year, chunk_start.month + 1, 1)

                            # Limit chunk_end to final end_date
                            if chunk_end > end_date:
                                chunk_end = end_date


                            chunk = filter_by_date(data, chunk_start, chunk_end)

                            if not chunk.empty:
                                process_days = detect_process_days(chunk)
                                list_process_days.extend(process_days.tolist())
                            
                            chunk_start = chunk_end # move to next

                         
                    st.success("✅ Process dates identified")
        
                except Exception as e:
                    st.error(f"Error getting process dates: {str(e)}")

                query = f"""
                SELECT m.timestamp, m.node_id, b.temperature, b.humidity, b.pressure, b.gas_resistance, s.co2
                FROM measurements m
                JOIN sensor_bme680 b ON m.id = b.measurement_id
                JOIN sensor_scd30 s ON m.id = s.measurement_id
                WHERE m.timestamp >= '{start_date}'
                AND m.timestamp <= '{end_date} 23:59:59'
                ORDER BY m.timestamp DESC
            
                """

                try:
                    with st.spinner("Loading data..."):
                        df = pd.read_sql_query(query, conn)
                        df = df.dropna()
                        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                        

                        list_process_days = pd.to_datetime(list_process_days)

                        list_process_days = [d.date() for d in list_process_days]

                        df = df[df['timestamp'].dt.date.isin(list_process_days)]

                        dt_series = pd.to_datetime(df['timestamp'])

                        # need to get these dates as process days output for the range between start_date and end_date
                        specific_dates = list_process_days
                        specific_dates = pd.to_datetime(specific_dates).date
                        mask_working_hours = (
                            (dt_series.dt.time >= pd.to_datetime("09:00").time()) & 
                            (dt_series.dt.time <= pd.to_datetime("18:00").time())
                        )
                        mask_specific_dates = dt_series.dt.date.isin(specific_dates)
                        combined_mask = mask_working_hours & mask_specific_dates

                        dataset = df[combined_mask] 


                        
                    st.success("✅ Data has been successfully loaded!")
            

                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
            

                if model_train_type == "Model A (Gas Resistance + Time)":
                    try:
                        with st.spinner("Preparing data..."):
                            dataset_gas_res = dataset[['node_id', 'timestamp', 'gas_resistance', 'co2']].copy()
                            cols_to_scale = ['co2']
                        
                            # Ensure only existing and numeric columns are scaled
                            cols_to_scale_present = [col for col in cols_to_scale if col in dataset_gas_res.columns]
                            dataset_gas_res[cols_to_scale_present] = dataset_gas_res[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                            dataset_gas_res[cols_to_scale_present] = dataset_gas_res[cols_to_scale_present] * 1e-2

                            # we will use it for model creation and while testing
                            with open("resources/A/used_node_ids.txt", "w") as f:
                                for node_id in nodes:
                                    f.write(f"{node_id}\n")


                        st.success("✅ Data has been successfully prepared!")

                    except Exception as e:
                        st.error(f"Error preparing data: {str(e)}")
                
                elif model_train_type == "Model B (All BME680 Features + Time)":
                    try:
                        with st.spinner("Preparing data..."):
                            dataset_gas_res = dataset[['node_id', 'timestamp', 'temperature', 'humidity', 'pressure', 'gas_resistance', 'co2']].copy()
                            cols_to_scale = ['co2', 'temperature', 'humidity', 'pressure']
                        
                            # Ensure only existing and numeric columns are scaled
                            cols_to_scale_present = [col for col in cols_to_scale if col in dataset_gas_res.columns]
                            dataset_gas_res[cols_to_scale_present] = dataset_gas_res[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                            dataset_gas_res[cols_to_scale_present] = dataset_gas_res[cols_to_scale_present] * 1e-2

                            # we will use it for model creation and while testing
                            with open("resources/B/used_node_ids.txt", "w") as f:
                                for node_id in nodes:
                                    f.write(f"{node_id}\n")

                        st.success("✅ Data has been successfully prepared!")

                    except Exception as e:
                        st.error(f"Error preparing data: {str(e)}")

                try:
                    with st.spinner("Feature engineering..."):
                        # Drop rows with missing values
                        dataset_gas_res.dropna(inplace=True)

                        # Convert timestamp and extract features
                        dataset_gas_res['timestamp'] = pd.to_datetime(dataset_gas_res['timestamp'])

                        # Extract raw time-based features
                        dataset_gas_res['hour'] = dataset_gas_res['timestamp'].dt.hour
                        dataset_gas_res['dayofweek'] = dataset_gas_res['timestamp'].dt.dayofweek
                        dataset_gas_res['month'] = dataset_gas_res['timestamp'].dt.month
                        dataset_gas_res['week_number'] = dataset_gas_res['timestamp'].dt.isocalendar().week


                        # Apply cyclic encoding for hour (0-23)
                        dataset_gas_res['hour_sin'], dataset_gas_res['hour_cos'] = cyclic_encode(dataset_gas_res['hour'], 24)

                        # Apply cyclic encoding for dayofweek (0-6)
                        dataset_gas_res['dayofweek_sin'], dataset_gas_res['dayofweek_cos'] = cyclic_encode(dataset_gas_res['dayofweek'], 7)

                        # Apply cyclic encoding for month (1-12)
                        dataset_gas_res['month_sin'], dataset_gas_res['month_cos'] = cyclic_encode(dataset_gas_res['month'], 12)

                        # Optionally, you can cyclic encode week_number as well (comment/uncomment as needed)
                        dataset_gas_res['week_number_sin'], dataset_gas_res['week_number_cos'] = cyclic_encode(dataset_gas_res['week_number'], 52)

                        # Drop original raw columns after cyclic encoding
                        dataset_gas_res.drop(columns=['hour', 'dayofweek', 'month'], inplace=True)
                        # Keep week_number numeric, or drop it if you want to use cyclic encoding

                        dataset_gas_res['node_id'] = dataset_gas_res['node_id'].astype(int) # typecasting

                        dataset_gas_res['node_id'] = pd.Categorical(
                            dataset_gas_res['node_id'],
                            categories = nodes
                        )

                        # One-hot encode 'node_id'
                        dataset_gas_res = pd.get_dummies(dataset_gas_res, columns=['node_id'], prefix='node')

                        # Drop timestamp (not used directly)
                        dataset_gas_res.drop(columns='timestamp', inplace=True)

                        if model_train_type == "Model A (Gas Resistance + Time)":
                            with open("resources/A/dataset_columns.txt", "w") as f:
                                for col in dataset_gas_res.columns:
                                    f.write(col + "\n")
                        elif model_train_type == "Model B (All BME680 Features + Time)":
                            with open("resources/B/dataset_columns.txt", "w") as f:
                                for col in dataset_gas_res.columns:
                                    f.write(col + "\n")




                    st.success("✅ Feature Engineering completed!")
                
                except Exception as e:
                        st.error(f"Error in Feature engineering: {str(e)}")


                
                try:
                    with st.spinner("Training model..."):
                        X = dataset_gas_res.drop(columns='co2')
                        y = dataset_gas_res['co2']

                        # train-test split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # feature scaling for quick learning
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # Define the optimized model
                        model_early_stop = Sequential([
                            Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
                            BatchNormalization(),
                            Dropout(0.3),

                            Dense(64, activation='relu'),
                            BatchNormalization(),
                            Dropout(0.3),

                            Dense(32, activation='relu'),
                            BatchNormalization(),
                            Dropout(0.2),

                            Dense(1)  # Output layer for regression
                        ])

                        model_early_stop.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

                        # Create the early stopping callback
                        early_stop = EarlyStopping(
                            monitor='val_loss',
                            patience=10,  # give more room for generalization
                            restore_best_weights=True
                        )

                        # Train the model with early stopping
                        history = model_early_stop.fit(
                            X_train_scaled, y_train,
                            epochs=num_of_epochs,  # enough epochs to let early stopping kick in      
                            batch_size=32,
                            validation_split=0.1,
                            verbose=1,
                            callbacks=[early_stop]
                        )

                        # evaluating as before
                        loss, mae = model_early_stop.evaluate(X_test_scaled, y_test)
                        print(f"Test MAE with Early Stopping: {mae:.2f}")
                    
                    st.success("✅ Model training completed!")

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")


                # Plot training and validation loss
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Train Loss')
                ax.plot(history.history['val_loss'], label='Val Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('MAE')
                ax.set_title('Training vs Validation Loss')
                ax.legend()
                ax.grid(True)

                # Show the plot in Streamlit
                st.pyplot(fig)

                try:
                    with st.spinner("Evaluating model on test split"):
                        y_pred = model_early_stop.predict(X_test_scaled)
                        y_pred_flat = y_pred.flatten()
                        results_df = pd.DataFrame({
                            'Actual_CO2': y_test.values,
                            'Predicted_CO2': y_pred_flat
                        }, index=y_test.index)
                        results_df.head()
                        results_df['5_Acutal_CO2_Lower'] = results_df['Actual_CO2'] * 0.95
                        results_df['5_Actual_CO2_Upper'] = results_df['Actual_CO2'] * 1.05
                        results_df['10_Acutal_CO2_Lower'] = results_df['Actual_CO2'] * 0.90
                        results_df['10_Actual_CO2_Upper'] = results_df['Actual_CO2'] * 1.10

                        results_df['5_Tolerable'] = (
                            (results_df['Predicted_CO2'] >= results_df['5_Acutal_CO2_Lower']) &
                            (results_df['Predicted_CO2'] <= results_df['5_Actual_CO2_Upper'])
                        )
                        results_df['10_Tolerable'] = (
                            (results_df['Predicted_CO2'] >= results_df['10_Acutal_CO2_Lower']) &
                            (results_df['Predicted_CO2'] <= results_df['10_Actual_CO2_Upper'])
                        )

                        accuracy_5 = results_df['5_Tolerable'].mean() * 100  # in percent
                        accuracy_10 = results_df['10_Tolerable'].mean() * 100  # in percent

                        st.metric("🔍 Accuracy within ±5% Tolerance", f"{accuracy_5:.2f}%")
                        st.metric("🔍 Accuracy within ±10% Tolerance", f"{accuracy_10:.2f}%")

                except Exception as e:
                    st.error(f"Evaluation on test split failed {str(e)}")


                

                if model_train_type == "Model A (Gas Resistance + Time)":
                    try:
                        with st.spinner("Saving scaler and model file"):
                            # Save the fitted scaler to a file
                            joblib.dump(scaler, 'models/A/scaler.pkl')
                            # saving the early stopping model
                            model_early_stop.save("models/A/model.keras")

                        st.success("✅ Model and scaler saved!")

                    except Exception as e:
                        st.error(f"Error saving scaler and model: {str(e)}")
                
                elif model_train_type == "Model B (All BME680 Features + Time)":
                    try:
                        with st.spinner("Saving scaler and model file"):
                            # Save the fitted scaler to a file
                            joblib.dump(scaler, 'models/B/scaler.pkl')
                            # saving the early stopping model
                            model_early_stop.save("models/B/model.keras")

                        st.success("✅ Model and scaler saved!")

                    except Exception as e:
                        st.error(f"Error saving scaler and model: {str(e)}")



                st.info("You can download the scaler and the model files below:")
                
                # Serialize the scaler to a bytes buffer
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                buffer.seek(0)  # Move to the beginning of the buffer

                # Download button for the serialized scaler
                st.download_button(
                    label="Download Scaler",
                    data=buffer,
                    file_name="scaler.pkl",
                    mime="application/octet-stream"
                )

                # Save the model to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                    model_early_stop.save(tmp.name)
                    tmp.seek(0)
                    buffer = io.BytesIO(tmp.read())

                st.download_button(
                    label="Download Model (.keras)",
                    data=buffer,
                    file_name="model.keras",
                    mime="application/octet-stream"
                )
                
                







            conn.close()
        
        except Exception as e:
            st.error(f"Database connection error: {str(e)}")
            if 'conn' in locals():
                conn.close() 

    elif mode == "Use the model you trained":
        st.subheader("📦 Choose the type of Model you trained")

        test_model_type = st.selectbox(
        "Select Test Model",
        ["-- Select Test Model --", "Model A (Gas Resistance + Time)", "Model B (All BME680 Features + Time)"]
        )

        if test_model_type == "Model A (Gas Resistance + Time)":
            st.info("🔄 Loading Model A...")
            try:
                with st.spinner("Loading Model A files..."):
                    model = load_model('models/A/model.keras')  # Keras model
                    scaler = joblib.load('models/A/scaler.pkl')  # Feature scaler
                st.success("Model A loaded successfully!")
            
            except Exception as e:
                st.error(f"Failed to load Model A: {e}")

            try:
                prediction_type = st.selectbox(
                    "Select How You Would Like To Test the Model",
                    ["-- Select Option --", "Entering Details for a Single Row", "Uploading a CSV file", "Selecting a Day from the Database"]
                )
                # Input fields
                with open("resources/A/used_node_ids.txt", "r") as f:
                    options = [line.strip() for line in f if line.strip()]

                # Read the column order from the file
                with open("resources/A/dataset_columns.txt", "r") as f:
                    expected_columns = [line.strip() for line in f if line.strip()]

            

                if prediction_type == "Entering Details for a Single Row": 
            
                    st.subheader("📝 Provide Input for Prediction")

                    options = sorted(options)

                    node_id = st.selectbox("Select Node ID", options=options, index=0)
                    st.subheader("📅 Enter Timestamp and Node Info")

                    # Ask for date and time separately
                    input_date = st.date_input("Select Date", value=date.today())
                    hour_text = st.text_input("Enter Hour (0–23)")
                    try:
                        hour = int(hour_text)
                        if not 0 <= hour <= 23:
                            st.error("Hour must be between 0 and 23.")
                            hour = None
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None
                
                    gas_resistance_text = st.text_input("Enter gas resistance.")
                    try:
                        gas_resistance = float(gas_resistance_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        gas_resistance = None


                    if st.button("📈 Predict CO₂ Level"):
                        if gas_resistance is None or input_date is None or node_id is None or hour is None:
                            st.warning("Please enter all fields.")
                        else:
                            try:
                                # Extract time features
                                week_number = input_date.isocalendar().week
                                hour = hour
                                dayofweek = input_date.weekday()
                                month = input_date.month

                                # Cyclic encodings
                                hour_sin, hour_cos = cyclic_encode(hour, 24)
                                dayofweek_sin, dayofweek_cos = cyclic_encode(dayofweek, 7)
                                month_sin, month_cos = cyclic_encode(month, 12)
                                week_sin, week_cos = cyclic_encode(week_number, 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in options]
                                node_encoding = {col: False for col in node_cols}
                                if f"node_{node_id}" in node_encoding:
                                    node_encoding[f"node_{node_id}"] = True

                                # Create final DataFrame row
                                input_dict = {
                                    "gas_resistance": gas_resistance,
                                    "week_number": week_number,
                                    "hour_sin": hour_sin,
                                    "hour_cos": hour_cos,
                                    "dayofweek_sin": dayofweek_sin,
                                    "dayofweek_cos": dayofweek_cos,
                                    "month_sin": month_sin,
                                    "month_cos": month_cos,
                                    "week_number_sin": week_sin,
                                    "week_number_cos": week_cos,
                                    **node_encoding
                                }

                                input_df = pd.DataFrame([input_dict])

                                                        
                                expected_columns = [col for col in expected_columns if col != 'co2']


                                input_df = input_df[expected_columns]

                            
                                # Scale input
                                scaled_input = scaler.transform(input_df)

                                # Predict
                                predicted_co2 = model.predict(scaled_input)[0][0]

                                st.success(f"🌿 Predicted CO₂ Level: **{predicted_co2:.2f} ppm**")
                        
                            except Exception as e:
                                st.error(f"❌ Single Row Prediction failed: {e}")

                elif prediction_type == "Uploading a CSV file":
                    st.subheader("📁 Upload CSV for Batch Prediction")
                    uploaded_file = st.file_uploader("Upload a CSV file with a 'timestamp', 'gas_resistance', and 'node_id' column", type=["csv"])
    
                    if uploaded_file:
                        try:
                            df = pd.read_csv(uploaded_file)

                            # Validate required columns
                            required_cols = {'timestamp', 'gas_resistance', 'node_id'}
                            if not required_cols.issubset(df.columns):
                                st.error(f"CSV must contain columns: {required_cols}")
                            else:
                                # Convert timestamp column to datetime
                                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                                df.dropna(subset=["timestamp", "gas_resistance", "node_id"], inplace=True)

                                # Extract time features
                                df["hour"] = df["timestamp"].dt.hour
                                df["dayofweek"] = df["timestamp"].dt.weekday
                                df["month"] = df["timestamp"].dt.month
                                df["week_number"] = df["timestamp"].dt.isocalendar().week

                                # Cyclic encodings
                                df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour"], 24)
                                df["dayofweek_sin"], df["dayofweek_cos"] = cyclic_encode(df["dayofweek"], 7)
                                df["month_sin"], df["month_cos"] = cyclic_encode(df["month"], 12)
                                df["week_number_sin"], df["week_number_cos"] = cyclic_encode(df["week_number"], 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in options]
                                for col in node_cols:
                                    df[col] = (df["node_id"] == int(col.split("_")[1])).astype(bool)

                            
                                expected_columns = [col for col in expected_columns if col != 'co2']

                                # Scale and predict
                                X = df[expected_columns]
                                X_scaled = scaler.transform(X)
                                predictions = model.predict(X_scaled).flatten()
                                df["predicted_co2"] = predictions

                                st.success("✅ Predictions completed!")
                                st.dataframe(df.head())

                                st.download_button(
                                    label="📥 Download CSV with Predictions",
                                    data=df.to_csv(index=False),
                                    file_name="predicted_co2.csv",
                                    mime="text/csv"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Failed to process file: {e}")

                elif prediction_type == "Selecting a Day from the Database":
                    st.subheader("📊 Compare Predicted vs Actual CO₂")

                    # User selects

                    if not os.path.exists(db_path):
                        st.error("Database file not found. Please check your .env configuration.")
                        return
        
                    try:
                        conn = sqlite3.connect(db_path)
        
                        # Get available date range
                        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)

                        selected_date = st.date_input(
                            "📅 Select Date",
                            value=default_start,
                            min_value=min_date,
                            max_value=max_date,
                            key="db_start_date"
                        )
                        selected_date = pd.to_datetime(selected_date)
                        start_dt = selected_date.strftime("%Y-%m-%d") + " 00:00:00"
                        end_dt = selected_date.strftime("%Y-%m-%d") + " 23:59:59"
                        selected_node = st.selectbox("Select Node ID", options=range(6, 13))
        


                    
                        query = f"""
                            SELECT m.timestamp, m.node_id, b.gas_resistance, s.co2
                            FROM measurements m
                            JOIN sensor_bme680 b ON m.id = b.measurement_id
                            JOIN sensor_scd30 s ON m.id = s.measurement_id
                            WHERE m.timestamp >= '{start_dt}'
                            AND m.timestamp <= '{end_dt}'
                            AND m.node_id = {selected_node}
                            ORDER BY m.timestamp
                            """
                        df_day = pd.read_sql_query(query,conn)
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day = df_day[(df_day["timestamp"].dt.hour >= 9) & (df_day["timestamp"].dt.hour < 18)]

                        if df_day.empty:
                            st.warning("No data found for the selected date and node.")
                            return
            
                        cols_to_scale = ['co2']
                        cols_to_scale_present = [col for col in cols_to_scale if col in df_day.columns]
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present] * 1e-2

                        # Feature engineering
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day["hour"] = df_day["timestamp"].dt.hour
                        df_day["dayofweek"] = df_day["timestamp"].dt.weekday
                        df_day["month"] = df_day["timestamp"].dt.month
                        df_day["week_number"] = df_day["timestamp"].dt.isocalendar().week

                        # Cyclic encodings
                        df_day["hour_sin"], df_day["hour_cos"] = cyclic_encode(df_day["hour"], 24)
                        df_day["dayofweek_sin"], df_day["dayofweek_cos"] = cyclic_encode(df_day["dayofweek"], 7)
                        df_day["month_sin"], df_day["month_cos"] = cyclic_encode(df_day["month"], 12)
                        df_day["week_number_sin"], df_day["week_number_cos"] = cyclic_encode(df_day["week_number"], 52)

                        # One-hot encode node
                        for i in options:
                            df_day[f"node_{i}"] = (df_day["node_id"] == i).astype(bool)

                        # Prepare feature matrix
                        expected_columns = [col for col in expected_columns if col != 'co2']

                        X = df_day[expected_columns]
                        X_scaled = scaler.transform(X)

                        # Predict
                        df_day["predicted_co2"] = model.predict(X_scaled).flatten()

                        # Plot
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(df_day["timestamp"], df_day["co2"], label="Actual CO₂", marker='o', linestyle='-')
                        ax.plot(df_day["timestamp"], df_day["predicted_co2"], label="Predicted CO₂", marker='x', linestyle='--')
                        ax.set_title(f"CO₂: Prediction vs Actual | Node {selected_node} | {selected_date.date()}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("CO₂ (ppm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"❌ Failed to generate plot: {e}")
                    finally:
                        if 'conn' in locals():
                            conn.close()


            except Exception as e:
                st.error(f"❌ Testing failed: {e}")

        elif test_model_type == "Model B (All BME680 Features + Time)":
            st.info("🔄 Loading Model B...")
            try:
                with st.spinner("Loading Model B files..."):
                    model = load_model('models/B/model.keras')
                    scaler = joblib.load('models/B/scaler.pkl')
                st.success("Model B loaded successfully!")
            
            except Exception as e:
                st.error(f"Failed to load Model B: {e}")

            try:
                prediction_type = st.selectbox(
                    "Select How You Would Like To Test the Model",
                    ["-- Select Option --", "Entering Details for a Single Row", "Uploading a CSV file", "Selecting a Day from the Database"]
                )
                # Input fields
                with open("resources/B/used_node_ids.txt", "r") as f:
                    options = [line.strip() for line in f if line.strip()]

                # Read the column order from the file
                with open("resources/B/dataset_columns.txt", "r") as f:
                    expected_columns = [line.strip() for line in f if line.strip()]

            

                if prediction_type == "Entering Details for a Single Row": 
            
                    st.subheader("📝 Provide Input for Prediction")

                    options = sorted(options)

                    node_id = st.selectbox("Select Node ID", options=options, index=0)
                    st.subheader("📅 Enter Timestamp and Node Info")

                    # Ask for date and time separately
                    input_date = st.date_input("Select Date", value=date.today())
                    hour_text = st.text_input("Enter Hour (0–23)")
                    try:
                        hour = int(hour_text)
                        if not 0 <= hour <= 23:
                            st.error("Hour must be between 0 and 23.")
                            hour = None
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None
                
                    gas_resistance_text = st.text_input("Enter gas resistance.")
                    try:
                        gas_resistance = float(gas_resistance_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        gas_resistance = None
                    
                    temperature_text = st.text_input("Enter temperature.")
                    try:
                        temperature_bme = float(temperature_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    humidity_text = st.text_input("Enter humidity.")
                    try:
                        humidity_bme = float(humidity_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    pressure_text = st.text_input("Enter pressure.")
                    try:
                        pressure = float(pressure_text)
                    except ValueError:
                        st.error("Please enter a valid number.")
                        hour = None


                    if st.button("📈 Predict CO₂ Level"):
                        if temperature_bme is None or humidity_bme is None or pressure is None or gas_resistance is None or input_date is None or node_id is None or hour is None:
                            st.warning("Please enter all fields.")
                        else:
                            try:
                                # Extract time features
                                week_number = input_date.isocalendar().week
                                hour = hour
                                dayofweek = input_date.weekday()
                                month = input_date.month

                                # Cyclic encodings
                                hour_sin, hour_cos = cyclic_encode(hour, 24)
                                dayofweek_sin, dayofweek_cos = cyclic_encode(dayofweek, 7)
                                month_sin, month_cos = cyclic_encode(month, 12)
                                week_sin, week_cos = cyclic_encode(week_number, 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in options]
                                node_encoding = {col: False for col in node_cols}
                                if f"node_{node_id}" in node_encoding:
                                    node_encoding[f"node_{node_id}"] = True

                                # Create final DataFrame row
                                input_dict = {
                                    "temperature": temperature_bme,
                                    "humidity": humidity_bme,
                                    "pressure": pressure,
                                    "gas_resistance": gas_resistance,
                                    "week_number": week_number,
                                    "hour_sin": hour_sin,
                                    "hour_cos": hour_cos,
                                    "dayofweek_sin": dayofweek_sin,
                                    "dayofweek_cos": dayofweek_cos,
                                    "month_sin": month_sin,
                                    "month_cos": month_cos,
                                    "week_number_sin": week_sin,
                                    "week_number_cos": week_cos,
                                    **node_encoding
                                }

                                input_df = pd.DataFrame([input_dict])

                                                        
                                expected_columns = [col for col in expected_columns if col != 'co2']


                                input_df = input_df[expected_columns]

                            
                                # Scale input
                                scaled_input = scaler.transform(input_df)

                                # Predict
                                predicted_co2 = model.predict(scaled_input)[0][0]

                                st.success(f"🌿 Predicted CO₂ Level: **{predicted_co2:.2f} ppm**")
                        
                            except Exception as e:
                                st.error(f"❌ Single Row Prediction failed: {e}")

                elif prediction_type == "Uploading a CSV file":
                    st.subheader("📁 Upload CSV for Batch Prediction")
                    uploaded_file = st.file_uploader("Upload a CSV file with a 'timestamp', 'temperature', 'humidity', 'pressure', 'gas_resistance', and 'node_id' column", type=["csv"])
    
                    if uploaded_file:
                        try:
                            df = pd.read_csv(uploaded_file)

                            # Validate required columns
                            required_cols = {'timestamp', 'temperature', 'humidity', 'pressure', 'gas_resistance', 'node_id'}
                            if not required_cols.issubset(df.columns):
                                st.error(f"CSV must contain columns: {required_cols}")
                            else:
                                # Convert timestamp column to datetime
                                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                                df.dropna(subset=["timestamp", "gas_resistance", "temperature", "humidity", "pressure", "node_id"], inplace=True)

                                # Extract time features
                                df["hour"] = df["timestamp"].dt.hour
                                df["dayofweek"] = df["timestamp"].dt.weekday
                                df["month"] = df["timestamp"].dt.month
                                df["week_number"] = df["timestamp"].dt.isocalendar().week

                                # Cyclic encodings
                                df["hour_sin"], df["hour_cos"] = cyclic_encode(df["hour"], 24)
                                df["dayofweek_sin"], df["dayofweek_cos"] = cyclic_encode(df["dayofweek"], 7)
                                df["month_sin"], df["month_cos"] = cyclic_encode(df["month"], 12)
                                df["week_number_sin"], df["week_number_cos"] = cyclic_encode(df["week_number"], 52)

                                # One-hot encoding for node_id
                                node_cols = [f"node_{i}" for i in options]
                                for col in node_cols:
                                    df[col] = (df["node_id"] == int(col.split("_")[1])).astype(bool)

                            
                                expected_columns = [col for col in expected_columns if col != 'co2']

                                # Scale and predict
                                X = df[expected_columns]
                                X_scaled = scaler.transform(X)
                                predictions = model.predict(X_scaled).flatten()
                                df["predicted_co2"] = predictions

                                st.success("✅ Predictions completed!")
                                st.dataframe(df.head())

                                st.download_button(
                                    label="📥 Download CSV with Predictions",
                                    data=df.to_csv(index=False),
                                    file_name="predicted_co2.csv",
                                    mime="text/csv"
                                )
                                
                        except Exception as e:
                            st.error(f"❌ Failed to process file: {e}")

                elif prediction_type == "Selecting a Day from the Database":
                    st.subheader("📊 Compare Predicted vs Actual CO₂")

                    # User selects

                    if not os.path.exists(db_path):
                        st.error("Database file not found. Please check your .env configuration.")
                        return
        
                    try:
                        conn = sqlite3.connect(db_path)
        
                        # Get available date range
                        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)

                        selected_date = st.date_input(
                            "📅 Select Date",
                            value=default_start,
                            min_value=min_date,
                            max_value=max_date,
                            key="db_start_date"
                        )
                        selected_date = pd.to_datetime(selected_date)
                        start_dt = selected_date.strftime("%Y-%m-%d") + " 00:00:00"
                        end_dt = selected_date.strftime("%Y-%m-%d") + " 23:59:59"
                        selected_node = st.selectbox("Select Node ID", options=range(6, 13))
        


                    
                        query = f"""
                            SELECT m.timestamp, m.node_id, b.temperature, b.humidity, b.pressure, b.gas_resistance, s.co2
                            FROM measurements m
                            JOIN sensor_bme680 b ON m.id = b.measurement_id
                            JOIN sensor_scd30 s ON m.id = s.measurement_id
                            WHERE m.timestamp >= '{start_dt}'
                            AND m.timestamp <= '{end_dt}'
                            AND m.node_id = {selected_node}
                            ORDER BY m.timestamp
                            """
                        df_day = pd.read_sql_query(query,conn)
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day = df_day[(df_day["timestamp"].dt.hour >= 9) & (df_day["timestamp"].dt.hour < 18)]

                        if df_day.empty:
                            st.warning("No data found for the selected date and node.")
                            return
            
                        cols_to_scale = ['temperature', 'humidity', 'pressure', 'co2']
                        cols_to_scale_present = [col for col in cols_to_scale if col in df_day.columns]
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present].apply(pd.to_numeric, errors='coerce')
                        df_day[cols_to_scale_present] = df_day[cols_to_scale_present] * 1e-2

                        # Feature engineering
                        df_day["timestamp"] = pd.to_datetime(df_day["timestamp"])
                        df_day["hour"] = df_day["timestamp"].dt.hour
                        df_day["dayofweek"] = df_day["timestamp"].dt.weekday
                        df_day["month"] = df_day["timestamp"].dt.month
                        df_day["week_number"] = df_day["timestamp"].dt.isocalendar().week

                        # Cyclic encodings
                        df_day["hour_sin"], df_day["hour_cos"] = cyclic_encode(df_day["hour"], 24)
                        df_day["dayofweek_sin"], df_day["dayofweek_cos"] = cyclic_encode(df_day["dayofweek"], 7)
                        df_day["month_sin"], df_day["month_cos"] = cyclic_encode(df_day["month"], 12)
                        df_day["week_number_sin"], df_day["week_number_cos"] = cyclic_encode(df_day["week_number"], 52)

                        # One-hot encode node
                        for i in options:
                            df_day[f"node_{i}"] = (df_day["node_id"] == i).astype(bool)

                        # Prepare feature matrix
                        expected_columns = [col for col in expected_columns if col != 'co2']

                        X = df_day[expected_columns]
                        X_scaled = scaler.transform(X)

                        # Predict
                        df_day["predicted_co2"] = model.predict(X_scaled).flatten()

                        # Plot
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(df_day["timestamp"], df_day["co2"], label="Actual CO₂", marker='o', linestyle='-')
                        ax.plot(df_day["timestamp"], df_day["predicted_co2"], label="Predicted CO₂", marker='x', linestyle='--')
                        ax.set_title(f"CO₂: Prediction vs Actual | Node {selected_node} | {selected_date.date()}")
                        ax.set_xlabel("Time")
                        ax.set_ylabel("CO₂ (ppm)")
                        ax.legend()
                        ax.grid(True)
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"❌ Failed to generate plot: {e}")
                    finally:
                        if 'conn' in locals():
                            conn.close()


            except Exception as e:
                st.error(f"❌ Testing failed: {e}")
                

        elif test_model_type == "-- Select Test Model --":
            st.info("Please select a test model to continue.")



    elif mode == "-- Select Mode --":
        st.info("Please select a mode to continue.")



    