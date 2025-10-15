#!/usr/bin/env python3
"""
Create Sample Database for SensorScope Analytics
Generates a sample SQLite database with sensor data for testing
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def create_sample_database(db_path='sample_sensor_data.db'):
    """Create a sample sensor database with multiple tables"""
    
    # Remove existing database if it exists
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Create temperature sensors table
    print("Creating temperature_sensors table...")
    temp_data = []
    start_date = datetime.now() - timedelta(days=30)
    
    for i in range(1000):
        timestamp = start_date + timedelta(hours=i*0.1)
        sensor_id = f"TEMP_{random.randint(1, 5):02d}"
        location = random.choice(['Lab_A', 'Lab_B', 'Factory_Floor', 'Storage', 'Office'])
        temperature = round(20 + 5 * np.sin(i * 0.1) + random.gauss(0, 2), 2)
        humidity = round(50 + 10 * np.cos(i * 0.05) + random.gauss(0, 5), 2)
        
        temp_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': sensor_id,
            'location': location,
            'temperature_c': temperature,
            'humidity_percent': max(0, min(100, humidity)),
            'status': random.choice(['active', 'active', 'active', 'maintenance'])
        })
    
    temp_df = pd.DataFrame(temp_data)
    temp_df.to_sql('temperature_sensors', conn, index=False, if_exists='replace')
    
    # Create pressure sensors table
    print("Creating pressure_sensors table...")
    pressure_data = []
    
    for i in range(800):
        timestamp = start_date + timedelta(hours=i*0.15)
        sensor_id = f"PRES_{random.randint(1, 3):02d}"
        location = random.choice(['Pump_Station_1', 'Pump_Station_2', 'Main_Line'])
        pressure = round(100 + 20 * np.sin(i * 0.2) + random.gauss(0, 5), 2)
        flow_rate = round(50 + pressure * 0.3 + random.gauss(0, 3), 2)
        
        pressure_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': sensor_id,
            'location': location,
            'pressure_bar': max(0, pressure),
            'flow_rate_lpm': max(0, flow_rate),
            'alert_level': 'normal' if pressure < 150 else 'high'
        })
    
    pressure_df = pd.DataFrame(pressure_data)
    pressure_df.to_sql('pressure_sensors', conn, index=False, if_exists='replace')
    
    # Create vibration sensors table
    print("Creating vibration_sensors table...")
    vib_data = []
    
    for i in range(600):
        timestamp = start_date + timedelta(hours=i*0.2)
        sensor_id = f"VIB_{random.randint(1, 4):02d}"
        equipment = random.choice(['Motor_A', 'Motor_B', 'Compressor_1', 'Fan_Unit'])
        vibration_x = round(random.gauss(0, 2), 3)
        vibration_y = round(random.gauss(0, 2), 3)
        vibration_z = round(random.gauss(0, 1.5), 3)
        frequency = round(50 + random.gauss(0, 5), 1)
        
        vib_data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'sensor_id': sensor_id,
            'equipment': equipment,
            'vibration_x_mm': vibration_x,
            'vibration_y_mm': vibration_y,
            'vibration_z_mm': vibration_z,
            'frequency_hz': frequency,
            'maintenance_due': random.choice([0, 0, 0, 1])  # 25% chance
        })
    
    vib_df = pd.DataFrame(vib_data)
    vib_df.to_sql('vibration_sensors', conn, index=False, if_exists='replace')
    
    # Create sensor metadata table
    print("Creating sensor_metadata table...")
    metadata = [
        {'sensor_id': 'TEMP_01', 'type': 'temperature', 'location': 'Lab_A', 'install_date': '2023-01-15', 'calibration_date': '2024-01-15'},
        {'sensor_id': 'TEMP_02', 'type': 'temperature', 'location': 'Lab_B', 'install_date': '2023-02-01', 'calibration_date': '2024-02-01'},
        {'sensor_id': 'TEMP_03', 'type': 'temperature', 'location': 'Factory_Floor', 'install_date': '2023-03-10', 'calibration_date': '2024-03-10'},
        {'sensor_id': 'PRES_01', 'type': 'pressure', 'location': 'Pump_Station_1', 'install_date': '2023-01-20', 'calibration_date': '2024-01-20'},
        {'sensor_id': 'PRES_02', 'type': 'pressure', 'location': 'Pump_Station_2', 'install_date': '2023-02-15', 'calibration_date': '2024-02-15'},
        {'sensor_id': 'VIB_01', 'type': 'vibration', 'location': 'Motor_A', 'install_date': '2023-01-25', 'calibration_date': '2024-01-25'},
        {'sensor_id': 'VIB_02', 'type': 'vibration', 'location': 'Motor_B', 'install_date': '2023-02-20', 'calibration_date': '2024-02-20'},
    ]
    
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_sql('sensor_metadata', conn, index=False, if_exists='replace')
    
    # Print summary
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print(f"\n✅ Sample database created: {db_path}")
    print(f"📊 Tables created: {len(tables)}")
    
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"   - {table_name}: {count} records")
    
    conn.close()
    
    # Create .env file if it doesn't exist
    env_path = '.env'
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(f"DATABASE_PATH={db_path}\n")
            f.write("DB_TIMEOUT=30\n")
            f.write("DB_CHECK_SAME_THREAD=False\n")
        print(f"✅ Created .env file pointing to {db_path}")
    else:
        print(f"⚠️ .env file already exists, not overwriting")
    
    print(f"\n🚀 Ready to test! Run: streamlit run app.py")

if __name__ == "__main__":
    create_sample_database() 