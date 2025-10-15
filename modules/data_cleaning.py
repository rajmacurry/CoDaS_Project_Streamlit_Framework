"""
Data Cleaning for the framework; ML and Semantically based cleaning
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

import joblib

import io
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

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
    





def show():
    st.header("🧹 Data Cleaning")
    st.markdown("Filtering Outliers, irregularities, null values, and duplicate records")

    db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
    if not os.path.exists(db_path):
        st.error("Database file not found. Please check your .env configuration.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [t[0] for t in cursor.fetchall()]

        st.subheader("🎯 Table Selection")
        table_options = [t for t in TABLES_SCHEMA.keys() if t in available_tables]
        selected_tables = st.multiselect("Choose one or two tables to explore:", table_options)

        if len(selected_tables) == 0:
            st.warning("Please select at least one table.")
            conn.close()
            return
        elif len(selected_tables) > 2:
            st.warning("Please select no more than two tables for merging.")
            conn.close()
            return

        st.subheader("📅 Date Filtering")
        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date:", value=default_start, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date:", value=default_end, min_value=min_date, max_value=max_date)

        if start_date > end_date:
            st.error("Start date must be before end date.")
            conn.close()
            return

        st.subheader("📊 Loaded Tables")

        try:
            with st.spinner("Loading data..."):
                dataframes = {}

                for table in selected_tables:
                    schema = TABLES_SCHEMA[table]
                    join_col = schema['join_col']
                    query = f"""
                        SELECT s.*, m.timestamp, m.node_id
                        FROM {table} s
                        JOIN measurements m ON s.{join_col} = m.id
                        WHERE m.timestamp >= '{start_date}'
                        AND m.timestamp <= '{end_date} 23:59:59'
                        ORDER BY m.timestamp DESC
                    """
                    df = pd.read_sql_query(query, conn)
                    dataframes[table] = df

                if len(selected_tables) >= 1:
                    df = merge_sensor_tables(selected_tables, dataframes)



                if df.empty:
                    st.warning(f"No data found in {selected_tables} for the selected date range.")
                else:
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Records Found", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Null Values", int(df.isna().sum().sum()))
                    with col4:
                        unique_nodes = df['node_id'].nunique() if 'node_id' in df.columns else 0
                        st.metric("Unique Nodes", unique_nodes)

                    available_nodes = sorted(df['node_id'].unique())

                    with st.expander("✅ Node Selection", expanded=True):
                        

                        

                        # Hardcoded default nodes
                        default_nodes = [9, 12, 10, 6, 8, 11, 7]

                        # Filter defaults to ensure they exist in the data
                        valid_default_nodes = [node for node in default_nodes if node in available_nodes]

                        # Sidebar multiselect
                        nodes = st.multiselect(
                            "Select node(s) to include",
                            options=available_nodes,
                            default=valid_default_nodes
                        )

                        if not nodes:
                            st.warning("Please select at least one node to continue.")
                        else:
                            #  Show all unique nodes in original data
                            available_nodes = df["node_id"].dropna().unique().tolist()
                            st.markdown(f"### 📍 Available Node IDs in Dataset: `{available_nodes}`")

                            #  Identify and report any dropped nodes
                            dropped_nodes = [node for node in available_nodes if node not in nodes]
                            if dropped_nodes:
                                st.warning(f"⚠️ The following node(s) were excluded based on your selection: `{dropped_nodes}`")
                                
                                # Extract excluded rows from the original dataframe
                                excluded_df = df[df['node_id'].isin(dropped_nodes)]
                                
                                if not excluded_df.empty:
                                    excluded_csv = excluded_df.to_csv(index=False)
                                    
                                    st.download_button(
                                        label="📥 Download Excluded Nodes Data",
                                        data=excluded_csv,
                                        file_name="excluded_nodes_data.csv",
                                        mime="text/csv"
                                    )
                            else:
                                st.info("No data rows found for excluded nodes to download.")

                            #  Filter dataframe to selected nodes
                            df = df[df['node_id'].isin(nodes)]

                            #  Confirm selected data
                            st.success(f"Showing data for selected node(s): `{nodes}`")

                            #  Correlate numeric features (excluding IDs and timestamps)
                            value = 0.01
                            excluded_cols = {'timestamp', 'measurement_id', 'local_id', 'id', 'node_id', 
                                             'local_id_sensor_scd30', 'local_id_sensor_bme680', 'localid_sps','localid_bme'}
                            candidate_features = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]

                            df[[f for f in candidate_features if f != "gas_resistance"]] = df[[f for f in candidate_features if f != "gas_resistance"]] * value

                            #  Display filtered and scaled data
                            st.subheader("🔗 Data")
                            st.write("Merged on `measurement_id` (renamed to `id`). Showing first 50 rows.")
                            st.dataframe(df.head(50))

                            # Parse timestamp if needed
                            df["timestamp"] = pd.to_datetime(df['timestamp'])



                    #  Show rows with any missing values
                    null_rows = df[df.isna().any(axis=1)]

                    st.markdown("""
                    ### 🕵️ Rows with Missing Values

                    The table below shows **rows that contain at least one null (missing) value**.
                    """)
                    st.dataframe(null_rows.head(100))

                    # 📊 Count of missing values per feature
                    st.markdown("##### 📉 Null Counts per Feature")
                    null_counts = df.isna().sum()
                    st.dataframe(null_counts[null_counts > 0].to_frame(name='null_count'))

                    # 📌 Count of null values per node (if 'node_id' exists)
                    if 'node_id' in df.columns:
                        st.markdown("##### 🧭 Null Counts per Node ID")
                        null_by_node = null_rows['node_id'].value_counts().to_frame(name='null_row_count')
                        st.dataframe(null_by_node)
                        plot_null_distributions(null_counts[null_counts > 0], null_by_node if 'node_id' in df.columns else None)
                    else:
                        st.info("Column `node_id` not found in data, so per-node null counts are unavailable.")

                    st.markdown("##### 🧹 Cleaning Null Values...")
                    df, null_df = clean_null_rows(df)

                    total_rows = len(df) + len(null_df)
                    removed_pct = (len(null_df) / total_rows) * 100 if total_rows > 0 else 0

                    st.success(
                        f"Removed {len(null_df)} rows containing null values "
                        f"({removed_pct:.2f}%). {len(df)} rows remaining."
                    )    



                    #  CSV download of cleaned data
                    with st.expander("⬇️ Download Cleaned & Removed Null Data", expanded=True):

                        # Generate CSV content as strings
                        cleaned_csv = df.to_csv(index=False)
                        null_csv = null_df.to_csv(index=False)

                        # Allow CSV download of cleaned and removed null rows

                        col1, col2 = st.columns(2)

                        with col1:
                            st.download_button(
                                label="📥 Download Cleaned Data",
                                data=cleaned_csv,
                                file_name="cleaned_data.csv",
                                mime="text/csv"
                            )

                        with col2:
                            st.download_button(
                                label="🗑️ Download Removed Null Rows",
                                data=null_csv,
                                file_name="null_rows.csv",
                                mime="text/csv"
                            )


                """
                    st.subheader("🧹 Duplicate Records Removal")
                    st.markdown(
                    This step identifies and removes *burst duplicate* sensor readings.  
                    These are consecutive records from the same node recorded within a short interval (default 10 seconds)  
                    where **all sensor values are identical**. This helps clean up noisy duplicate data and ensures  
                    each measurement is unique per node and time.
                    )

                # Call the function with your dataframe (df) and sensor columns
                df, duplicates_df = filter_burst_duplicates(df, candidate_features, time_interval_sec=10)

                st.success(f"Removed {len(duplicates_df)} duplicate rows. Remaining rows: {len(df)}.")
                st.dataframe(duplicates_df.head(100))
                """

                    
                # --- Outlier Filtering Section ---
                st.markdown("## 🧹 Outlier Filtering")
                st.markdown("""
                This step removes extreme values that may be caused by sensor errors, environmental noise, or transmission faults.
                You can choose from the following methods:
                - **Hardcoded Bounds**: Uses fixed min/max thresholds based on sensor specs.
                - **IQR Filtering**: Uses statistical interquartile range to filter extreme values.
                - **SVM-Based Filtering**: Uses a trained model (enabled only if both SCD and BME680 are present).
                """)

                # Force IQR if only SPS30 is selected
                force_iqr = "sps30" in selected_tables and len(selected_tables) == 1
                has_scd = any("scd" in tbl.lower() for tbl in selected_tables)
                has_bme = any("bme" in tbl.lower() for tbl in selected_tables)
                has_sps = any("sps" in tbl.lower() for tbl in selected_tables)
                # Determine allowed methods
                # Determine available filtering methods
                available_methods = ["Hardcoded Bounds", "IQR Filtering"]
                if has_scd and has_bme:
                    available_methods.append("Hardcoded Bounds + SVM")

                # --- UI ---
                if force_iqr:
                    st.info("⚠️ Only SPS30 table selected — defaulting to **IQR Filtering**.")
                    selected_method = "IQR Filtering"
                else:
                    selected_method = st.radio("### Select Outlier Filtering Method:", available_methods)

                # Apply filtering based on selected method
                if selected_method == "Hardcoded Bounds":
                    st.markdown("#### 🚧 Applying Hardcoded Bounds Filtering")
                    # df = apply_hardcoded_bounds(df)
                    # Define hardcoded bounds
                    scd_bounds = {
                        'co2': (100, 1000),
                        'temperature': (12, 35),
                        'humidity': (14, 80)
                    }

                    bme_bounds = {
                        'temperature_bme': (12, 35),
                        'humidity_bme': (10.45, 61.45),
                        'pressure': (61521, 103777),
                        'gas_resistance': (3000, 5430598)
                    }

                    selected_lower_bounds = {}
                    selected_upper_bounds = {}

                    if len(selected_tables) == 1:
                        table = selected_tables[0].lower()
                        if "scd" in table:
                            default_bounds = scd_bounds
                        elif "bme" in table:
                            default_bounds = bme_bounds
                        else:
                            st.warning("No hardcoded bounds defined for selected table.")
                            default_bounds = {}

                        with st.form("Edit Bounds"):
                            st.markdown("#### ✏️ Edit Bounds for Selected Table")
                            for feature, (low, high) in default_bounds.items():
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col1:
                                    st.markdown(f"**{feature}**")
                                with col2:
                                    new_low = st.number_input(f"Min {feature}", value=float(low), key=f"min_{feature}")
                                with col3:
                                    new_high = st.number_input(f"Max {feature}", value=float(high), key=f"max_{feature}")
                                selected_lower_bounds[feature] = new_low
                                selected_upper_bounds[feature] = new_high

                            submitted = st.form_submit_button("Apply Bounds")

                        if submitted:
                            user_bounds = {f: (selected_lower_bounds[f], selected_upper_bounds[f]) for f in selected_lower_bounds}
                            df, extreme_outliers = apply_bounds_filter(df, user_bounds)

                            st.success("✅ Hardcoded bounds applied. Cleaned and outlier data generated.")

                            with st.expander("📈 View Summary"):
                                st.markdown(f"**Cleaned Data Size:** {len(df)}")
                                st.markdown(f"**Outlier Data Size:** {len(extreme_outliers)}")
                                st.dataframe(extreme_outliers.head())
                    
                    elif len(selected_tables) > 1:
                        st.info("Multiple tables selected. Bounds editing is currently supported for one table at a time.")
                    else:
                        st.warning("Please select at least one table to apply hardcoded bounds.")




                elif selected_method == "IQR Filtering":
                    st.markdown("#### 📈 Applying IQR-Based Outlier Filtering")
                    st.markdown("Use the sliders below to set the lower and upper percentile thresholds for filtering extreme outliers based on IQR.")

                    lower_percentile = st.slider(
                        "Select lower percentile bound for IQR filtering",
                        min_value=0.0,
                        max_value=0.4,
                        value=0.02,
                        step=0.01,
                        format="%.2f"
                    )

                    upper_percentile = st.slider(
                        "Select upper percentile bound for IQR filtering",
                        min_value=0.6,
                        max_value=1.0,
                        value=0.98,
                        step=0.01,
                        format="%.2f"
                    )

                    if lower_percentile >= upper_percentile:
                        st.error("Lower percentile must be less than upper percentile.")
                    else:
                        # Ask the user for the IQR factor
                        factor = st.number_input(
                            "Enter the IQR factor for outlier detection (typically between 1.5 and 3.0):",
                            min_value=0.0,
                            max_value=10.0,
                            value=2.0,
                            step=0.1
                        )

                        df, extreme_outliers = filter_extreme_iqr_outliers(
                            df, candidate_features,
                            factor=factor,
                            lower_percentile=lower_percentile,
                            upper_percentile=upper_percentile
                        )
                        st.success(f"Removed {len(extreme_outliers)} extreme outliers using IQR filtering.")
                        st.write(f"{len(df)} rows remaining after filtering.")



                elif selected_method == "Hardcoded Bounds + SVM":
                    st.markdown("#### 🤖 Applying Hardcoded Bounds + SVM Filtering")
                    features = [
                            'co2', 'temperature', 'humidity', 
                            'temperature_bme', 'humidity_bme', 
                            'pressure', 'gas_resistance'
                        ]
                    extreme_outliers, df = flag_outliers(df, features)
                    



                if not extreme_outliers.empty:
                    st.markdown("""
                        ##### 🧪 Removed Outlier Data

                        The following rows were identified as **outliers** based on the applied data cleaning strategy.  
                        Outliers are defined as values falling outside the acceptable bounds or thresholds for selected features.  
                        This step helps eliminate sensor glitches, anomalies, or extreme deviations that may affect analysis quality.

                        You can preview the top 100 removed rows and see summary statistics (mean, std, min, max, etc.) of the excluded data.
                        """)
                    st.write(f"Showing {len(extreme_outliers)} removed rows:")
                    st.dataframe(extreme_outliers.head(100))
                    st.markdown("""
                    The following dataframe discribe the characteristics of the features which are considered outliers. """)
                    st.dataframe(extreme_outliers.describe())
                else:
                    st.info("✅ No extreme outliers detected for the selected features.")


                if not extreme_outliers.empty:
                    with st.expander("⬇️ Download Cleaned or Outlier Data", expanded=True):

                        # Generate CSV content as strings
                        cleaned_csv = df.to_csv(index=False)
                        extreme_csv = extreme_outliers.to_csv(index=False)

                        # Allow CSV download of cleaned and removed null rows

                        col1, col2 = st.columns(2)

                        with col1:
                            st.download_button(
                                label="📥 Download Outlier Cleaned Data",
                                data=cleaned_csv,
                                file_name="cleaned_data.csv",
                                mime="text/csv"
                            )

                        with col2:
                            st.download_button(
                                label="🗑️ Download Outliers Data",
                                data=extreme_csv,
                                file_name="outlier.csv",
                                mime="text/csv"
                            )


                




        except Exception as e:
            st.error(f"Error loading or merging data: {str(e)}")

        finally:
            conn.close()

    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        if 'conn' in locals():
            conn.close()
























def merge_sensor_tables(selected_tables, dataframes):
    """
    Merge multiple sensor tables based on measurement_id (renamed to 'id').
    Keeps only one 'timestamp' and 'node_id' column from the first table.
    """
    if not selected_tables:
        return None

    # Renaming rules for known tables
    rename_rules = {
            "sensor_scd30": {
                "id": "local_id",
                "measurement_id": "id",
                "humidity": "humidity",
                "temperature": "temperature",
                "co2": "co2"
            },
            "sensor_bme680": {
                "measurement_id": "id",
                "co2": "co2_bme",
                "humidity": "humidity_bme",
                "temperature": "temperature_bme",
                "id": "localid_bme"
            },
            "sensor_sps30": {
                "measurement_id": "id",
                "id": "localid_sps",
            },
        }

    base_table = selected_tables[0]
    data = dataframes[base_table].copy()

    # Apply renaming to base table
    if base_table in rename_rules:
        data = data.rename(columns=rename_rules[base_table])
    else:
        data = data.rename(columns={"measurement_id": "id", "id": f"local_id_{base_table}"})

    # Track timestamp/node_id source
    authoritative_timestamp = "timestamp"
    authoritative_node_id = "node_id"

    for table in selected_tables[1:]:
        df = dataframes[table].copy()

        # Apply renaming
        if table in rename_rules:
            df = df.rename(columns=rename_rules[table])
        else:
            df = df.rename(columns={"measurement_id": "id", "id": f"local_id_{table}"})

        # Drop extra timestamp and node_id if they exist
        df = df.drop(columns=[col for col in ["timestamp", "node_id"] if col in df.columns])

        # Merge on id
        data = pd.merge(data, df, on="id", how="outer")

    # Ensure 'id', 'timestamp', 'node_id' are first
    id_cols = ["id", "timestamp", "node_id"]
    other_cols = [col for col in data.columns if col not in id_cols]
    data = data[id_cols + other_cols if "timestamp" in data.columns and "node_id" in data.columns else data.columns]

    return data



def clean_null_rows(df):
    """
    Remove rows with any null values from df.
    Returns: cleaned_df, removed_null_rows_df
    """
    null_df = df[df.isna().any(axis=1)].copy()
    cleaned_df = df.dropna().copy()
    return cleaned_df, null_df





def plot_null_distributions(null_counts, null_by_node=None):
    """
    Plots bar charts for null counts per feature and optionally per node_id.
    
    Parameters:
    - null_counts: pd.Series of null count per column (features)
    - null_by_node: pd.Series or pd.DataFrame with null row counts per node_id (optional)
    """
    # 📉 Plot: Nulls per Feature
    st.markdown("#### 📊 Missing Values per Feature (Bar Plot)")
    if not null_counts.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=null_counts.index, y=null_counts.values, ax=ax1, palette="Reds_r")
        ax1.set_ylabel("Null Count")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha="right")
        ax1.set_title("Null Counts per Feature")
        st.pyplot(fig1)
    else:
        st.info("✅ No missing values found in features.")

    # 🧭 Plot: Nulls per Node
    if null_by_node is not None and not null_by_node.empty:
        st.markdown("#### 🧭 Missing Rows per Node ID (Bar Plot)")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if isinstance(null_by_node, pd.DataFrame):
            data = null_by_node.reset_index()
            data.columns = ['node_id', 'null_row_count']
        else:
            data = null_by_node.reset_index()
            data.columns = ['node_id', 'null_row_count']
        sns.barplot(x='node_id', y='null_row_count', data=data, ax=ax2, palette="Blues_d")
        ax2.set_ylabel("Rows with Missing Values")
        ax2.set_title("Null Row Count per Node ID")
        st.pyplot(fig2)
    elif null_by_node is not None:
        st.info("✅ No missing rows per node found.")



def filter_burst_duplicates(df, sensor_columns=None, time_interval_sec=10):
    """
    Remove burst duplicates per node based only on timestamp proximity (ignore sensor values).
    
    Keeps the first record per node and removes subsequent records within `time_interval_sec`.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with 'timestamp' (datetime) and 'node_id'.
        sensor_columns (list of str): Not used here but kept for API compatibility.
        time_interval_sec (int or float): Time interval in seconds to consider for duplicates.
        
    Returns:
        deduped_df (pd.DataFrame): DataFrame with duplicates removed.
        duplicates_df (pd.DataFrame): DataFrame with only duplicate rows.
    """

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['node_id', 'timestamp']).reset_index(drop=True)

    # Shift previous row info per node
    df['prev_node_id'] = df['node_id'].shift(1)
    df['prev_timestamp'] = df['timestamp'].shift(1)

    # Calculate time difference in seconds from previous record
    df['time_diff'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()

    # Identify burst duplicates: same node, time_diff < interval
    df['is_burst_duplicate'] = (
        (df['node_id'] == df['prev_node_id']) &
        (df['time_diff'] < time_interval_sec)
    )

    deduped_df = df[~df['is_burst_duplicate']].copy()

    # Clean up helper columns
    drop_cols = ['prev_node_id', 'prev_timestamp', 'time_diff', 'is_burst_duplicate']
    deduped_df.drop(columns=drop_cols, inplace=True)

    duplicates_df = df[df['is_burst_duplicate']].copy()
    duplicates_df.drop(columns=drop_cols, inplace=True)

    return deduped_df, duplicates_df




def filter_extreme_iqr_outliers(df, candidate_features, factor=2.0, lower_percentile=0.2, upper_percentile=0.98):
    """
    Remove extreme outliers from the DataFrame using IQR filtering.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        candidate_features (list): List of numeric columns to apply IQR filtering on.
        factor (float): Multiplier for the IQR to define extreme outlier range (default = 3.0).

    Returns:
        filtered_df (pd.DataFrame): DataFrame with outliers removed.
        outlier_df (pd.DataFrame): DataFrame containing the removed outlier rows.
    """
    df = df.copy()
    outlier_mask = pd.Series(False, index=df.index)

    for col in candidate_features:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        Q1 = df[col].quantile(lower_percentile)
        Q3 = df[col].quantile(upper_percentile)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        outlier_mask |= col_outliers

    outlier_df = df[outlier_mask].copy()
    filtered_df = df[~outlier_mask].copy()

    return filtered_df, outlier_df



def flag_outliers(data, features, id_col='measurement_id', model_path=os.path.join(os.path.dirname(__file__), 'svm_outlier_model.pkl')):
    st.markdown("#### Outlier Detection with Pretrained SVM Model")
    st.markdown("""
    This function:
    - Loads a pretrained SVM model pipeline (including scaler and classifier).
    - Applies the model to the new dataset's features to predict outliers.
    - Flags readings predicted as outliers.
    - Displays diagnostic statistics about the SVM model (support vectors, coefficients, decision margins).
    """)

    # Load model
    model = joblib.load(model_path)
    st.markdown(f"### Loaded model")

    # Predict outliers
    X_new = data[features]
    data = data.copy()  # avoid modifying the original DataFrame
    predictions = model.predict(X_new)

    # Add temporary column
    data['_outlier_predicted'] = predictions

    st.markdown(f"### Prediction complete")
    st.write(f"Total records flagged as outliers: **{(predictions == 1).sum()}**")

    # Extract SVC model from pipeline
    svc = model.named_steps['svc']

    # Diagnostics
    st.markdown("### SVM Model Diagnostics")
    st.markdown("- **Number of support vectors per class:**")
    st.write(svc.n_support_)
    st.markdown("- **Total number of support vectors:**")
    st.write(svc.support_vectors_.shape[0])
    st.markdown("- **Shape of support vectors matrix:**")
    st.write(svc.support_vectors_.shape)
    st.markdown("- **Intercept term (bias):**")
    st.write(svc.intercept_)
    st.markdown("- **Shape of dual coefficients matrix:**")
    st.write(svc.dual_coef_.shape)

    # Decision function statistics
    decision_values = model.decision_function(X_new)
    data['margin'] = decision_values
    st.markdown("### Decision Function Statistics (distance from decision boundary)")
    st.write({
        "Min margin": decision_values.min(),
        "Mean margin": decision_values.mean(),
        "Max margin": decision_values.max()
    })


    #Calling the plotting function
    plot_decision_boundary_from_data(data, model, features, label_col='_outlier_predicted')

    # Separate flagged and cleaned, and drop the internal column
    flagged = data[data['_outlier_predicted'] == 1].drop(columns=['_outlier_predicted'])
    cleaned = data[data['_outlier_predicted'] == 0].drop(columns=['_outlier_predicted'])

    return flagged, cleaned



def apply_bounds_filter(df, bounds):
    """
    Filters out rows with out-of-bound values based on provided bounds.

    Parameters:
        df (pd.DataFrame): Input dataframe
        bounds (dict): Dictionary with {column_name: (min, max)}

    Returns:
        cleaned_df (pd.DataFrame): Data within bounds
        outliers_df (pd.DataFrame): Data outside bounds
    """
    mask = pd.Series(True, index=df.index)
    for col, (lower, upper) in bounds.items():
        if col in df.columns:
            mask &= df[col].between(lower, upper)
    cleaned_df = df[mask]
    outliers_df = df[~mask]
    return cleaned_df, outliers_df





def plot_decision_boundary_from_data(data, model, features, label_col='outlier_predicted'):
    st.markdown("##### 🧠 SVM Decision Boundary Visualization")

    st.markdown("""
    This plot illustrates how the trained SVM model separates outliers from normal readings based on the selected **two features**.

    **What this visualization shows:**
    - The **colored background** represents the decision function values of the SVM classifier.
      - The **boundary** (where decision function = 0) separates predicted outliers from clean readings.
      - **Red and blue hues** show model confidence in prediction regions.
    - The **data points** are plotted with their actual prediction labels.
    - You can **select any two features** below to see how they contribute to the decision boundary.

    **Note**: All other model features are fixed at their mean values to allow a 2D projection.
    """)

    # Feature selection (2D)
    selected_feats = st.multiselect("Select 2 features to visualize", features, default=features[:2], max_selections=2)
    if len(selected_feats) != 2:
        st.warning("Please select exactly 2 features to visualize the decision boundary.")
        return

    feat_x, feat_y = selected_feats

    # Get scaler and SVC from pipeline
    scaler = model.named_steps['standardscaler']
    svc = model.named_steps['svc']

    # Create mesh grid for plotting
    x_min, x_max = data[feat_x].min() - 1, data[feat_x].max() + 1
    y_min, y_max = data[feat_y].min() - 1, data[feat_y].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Create full feature matrix with mean values
    X_grid = pd.DataFrame(np.tile(data[features].mean().values, (grid.shape[0], 1)), columns=features)
    X_grid[feat_x] = grid[:, 0]
    X_grid[feat_y] = grid[:, 1]

    # Standardize and compute decision function
    X_grid_std = scaler.transform(X_grid)
    Z = svc.decision_function(X_grid_std).reshape(xx.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(xx, yy, Z, levels=20, cmap="coolwarm", alpha=0.6)
    fig.colorbar(contour, ax=ax, label='Decision function')

    sns.scatterplot(data=data, x=feat_x, y=feat_y, hue=label_col, palette="Set1", edgecolor='k', ax=ax)

    ax.set_title("SVM Decision Boundary")
    ax.set_xlabel(feat_x)
    ax.set_ylabel(feat_y)
    ax.grid(True)

    st.pyplot(fig)