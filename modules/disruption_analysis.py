"""
Disruption Analysis; Consisting of node level, network level disruption and irregular reading
"""

import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates


import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    """Get appropriate datetime range for measurements table"""
    try:
        # Get data range from measurements table
        date_query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM measurements"
        date_df = pd.read_sql_query(date_query, conn)
        
        if not date_df.empty and date_df['min_date'].iloc[0]:
            min_date = pd.to_datetime(date_df['min_date'].iloc[0])
            max_date = pd.to_datetime(date_df['max_date'].iloc[0])
            
            # Default to last month of available data
            default_start = max_date - timedelta(days=default_days)
            if default_start < min_date:
                default_start = min_date
                
            return min_date, max_date, default_start, max_date
        else:
            # Fallback for empty data - use 2021-2022 range
            return datetime(2021, 1, 1), datetime(2022, 12, 31), datetime(2022, 12, 1), datetime(2022, 12, 31)
    except:
        # Fallback dates for 2021-2022
        return datetime(2021, 1, 1), datetime(2022, 12, 31), datetime(2022, 12, 1), datetime(2022, 12, 31)
    

def show():
    st.header("❌ Disruption Analysis")
    st.markdown("Showcasing disruptions and, irregularities on different granularity level")
    
    db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
    
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
        
        # Table selection
        st.subheader("🎯 Table Selection")
        table_options = [t for t in TABLES_SCHEMA.keys() if t in available_tables]
        
        if not table_options:
            st.error("Expected sensor tables not found in database.")
            conn.close()
            return
        
        selected_table = st.selectbox("Choose table to explore:", table_options)

        # DateTime filtering for all tables (via measurements join)
        st.subheader("📅 DateTime Filtering")
        
        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Start DateTime:**")
            start_date_part = st.date_input(
                "Start Date:",
                value=default_start.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="db_start_date"
            )
            start_time_part = st.time_input(
                "Start Time:",
                value=default_start.time(),
                key="db_start_time"
            )
            start_date = datetime.combine(start_date_part, start_time_part)
        
        with col2:
            st.markdown("**End DateTime:**")
            end_date_part = st.date_input(
                "End Date:",
                value=default_end.date(),
                min_value=min_date.date(),
                max_value=max_date.date(),
                key="db_end_date"
            )
            end_time_part = st.time_input(
                "End Time:",
                value=default_end.time(),
                key="db_end_time"
            )
            end_date = datetime.combine(end_date_part, end_time_part)
        
        # Validate datetime range
        if start_date > end_date:
            st.error("Start datetime must be before end datetime.")
            conn.close()
            return
        
        date_diff = (end_date - start_date).days
        if date_diff > 31:
            st.warning(f"⚠️ DateTime range is {date_diff} days. Consider a shorter range for better performance.")
        # Build query based on selected table
        if selected_table == 'measurements':
            query = f"""
            SELECT * FROM measurements 
            WHERE timestamp >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}' 
            AND timestamp <= '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY timestamp DESC
            
            """
            # LIMIT 10000
        else:
            # Join with measurements table for date filtering
            schema = TABLES_SCHEMA[selected_table]
            join_col = schema['join_col']
            
            query = f"""
            SELECT s.*, m.timestamp, m.node_id
            FROM {selected_table} s
            JOIN measurements m ON s.{join_col} = m.id
            WHERE m.timestamp >= '{start_date.strftime('%Y-%m-%d %H:%M:%S')}'
            AND m.timestamp <= '{end_date.strftime('%Y-%m-%d %H:%M:%S')}'
            ORDER BY m.timestamp DESC
            
            """
            #LIMIT 10000
        
        # Execute query and display results
        st.subheader(f"{selected_table.title()}")
        
        try:
            with st.spinner("Loading data..."):
                df = pd.read_sql_query(query, conn)
                # Get unique node IDs from the data
            

            if df.empty:
                st.warning(f"No data found in {selected_table} for the selected date range.")
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

                    # Warn if no nodes selected
                    if not nodes:
                        st.warning("Please select at least one node to continue.")
                    else:
                        # Filter dataframe
                        df = df[df['node_id'].isin(nodes)]

                        # Preview data
                        st.write(f"Showing data for nodes: {nodes}")
                        # To do the correlation of 10^-2 before describe of the data
                        value = 0.01
                        excluded_cols = {'timestamp', 'measurement_id', 'local_id', 'id', 'node_id'}
                        candidate_features = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]

                        df[[f for f in candidate_features if f != "gas_resistance"]] = df[[f for f in candidate_features if f != "gas_resistance"]] * value
                        st.dataframe(df.head())

                    
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
  
                st.markdown("""
                    ### 📊❌ Recorded vs Missing Sensor Readings

                    The plot below compares **expected vs actual sensor readings** for each node over time.

                    - **Green dots** represent timestamps where a reading was successfully recorded.
                    - **Red dots** indicate missing readings, where no data was received within the expected interval (±10 seconds).
                    - Each horizontal line corresponds to a different node.

                    Use this visualization to detect patterns of data loss, communication issues, or node-specific outages.
                    """)
                
                # User input for interval and jitter tolerance
                with st.expander("⚙️ Reading Interval and Tolerance Settings", expanded=False):
                    interval_seconds = st.number_input(
                        "Expected Interval Between Readings (seconds)",
                        min_value=0.000001,
                        max_value=3600.0,
                        value=10.0,
                        step=0.1,
                        format="%.6f",
                        help="Expected time between readings. Supports microsecond resolution."
                    )

                    jitter_tolerance_seconds = st.number_input(
                        "Jitter Tolerance (seconds)",
                        min_value=0.0,
                        max_value=60.0,
                        value=10.0,
                        step=0.1,
                        format="%.6f",
                        help="Allowed deviation from expected interval before a reading is considered missing."
                    )

                # Convert to timedelta if needed
                interval = pd.to_timedelta(interval_seconds, unit='s')
                jitter = pd.to_timedelta(jitter_tolerance_seconds, unit='s')

                # Call function with float seconds or convert accordingly
                result = check_expected_vs_actual(
                    df, start_date, end_date,
                    interval_seconds=interval.total_seconds(),
                    jitter_tolerance_seconds=jitter.total_seconds()
                )

                plot_recorded_vs_missing(result)

                st.markdown("""
                #### 📈 Uptime Summary per Node

                This table shows the **recorded vs missing data points** for each sensor node, along with the calculated **uptime percentage**.

                - **recorded**: Number of expected timestamps where data was received within the allowed time window.
                - **missing**: Number of expected timestamps with no corresponding data received.
                - **total**: Total expected readings in the time range.
                - **uptime_%**: Percentage of time the node successfully reported data.

                Use this summary to identify nodes with persistent communication issues or low reliability.
                """)

                uptime_summary = calculate_uptime_summary(result)

                uptime_summary['uptime_%'] = uptime_summary['uptime_%'].map(lambda x: f"{x:.2f}")
                st.dataframe(uptime_summary)
                st.markdown("""
                    #### 📶 Node Uptime Overview

                    The bar chart below shows the **uptime percentage** for each sensor node over the selected time range.

                    - 100% means the node reported every expected reading.
                    - Lower percentages may indicate communication issues or node outages.

                    Use this visualization to quickly identify underperforming nodes.
                    """)
                plot_uptime_bar(uptime_summary)

                # Streamlit widgets for user input
                st.subheader("📟❌ Node Disruption Analysis")  
                st.markdown("""
                    #### ⚙️ Disruption Detection Settings

                    Adjust the settings below to control how disruptions are detected:

                    - **Expected Interval**: How often each node is supposed to report data (e.g., every 10 seconds).
                    - **Minimum Disruption Duration**: The minimum length of a missing-data period to count as a disruption (e.g., 10 minutes).

                    These values help filter out brief or expected communication gaps and focus on meaningful outages.
                    """)
                expected_interval_sec = st.number_input("Expected interval between readings (seconds):", min_value=1, value=10)
                min_disruption_duration = st.text_input("Minimum disruption duration (e.g., '5min', '1h'):", value='10min')

                # Convert to Timedelta
                expected_interval = pd.Timedelta(seconds=expected_interval_sec)
                min_duration = min_disruption_duration  # string is okay; function handles conversion

                # Find disruptions
                disruptions_df = find_disruptions(result, expected_interval=expected_interval, min_duration=min_duration)
                st.markdown("""
                    ### ⚠️ Detected Disruptions

                    Below is a list of **data disruptions** detected for each node based on the selected criteria.

                    - Each row represents a continuous period of missing data.
                    - Only gaps **longer than the minimum disruption duration** are shown.
                    - Use this to monitor network reliability and identify problematic nodes.

                    """)

                if disruptions_df.empty:
                    st.info("✅ No disruptions found.")
                else:
                    disruptions_df['duration'] = disruptions_df['duration'].apply(
                            lambda d: str(pd.to_timedelta(d)).split('.')[0]  # Remove microseconds
                        )
                    st.dataframe(disruptions_df)
                    st.markdown("""
                        ### 📏 Disruption Grouping Thresholds

                        Specify the duration thresholds to group disruptions into categories.

                        - Enter durations using units like **seconds (s)**, **minutes (min)**, **hours (h)**, **days (D)**, or **weeks (W)**.
                        - Examples:
                        - `"30min"` for 30 minutes
                        - `"1h"` or `"1hour"` for 1 hour
                        - `"1D"` or `"1day"` for 1 day
                        - `"7D"` for 7 days (1 week)

                        The disruptions will be grouped into ranges based on these thresholds, helping you identify short, medium, and long outages.

                        Make sure to enter valid time strings compatible with pandas Timedelta.
                        """)
                    t1 = st.text_input("Threshold 1 (e.g., '1D')", value="1D")
                    t2 = st.text_input("Threshold 2 (e.g., '7D')", value="7D")
                    t3 = st.text_input("Threshold 3 (e.g., '30D')", value="30D")

                    thresholds = [t1, t2, t3]

                    try:
                        thresholds = [pd.Timedelta(t) for t in [t1, t2, t3]]
                        disruptions_grouped = group_disruptions_by_duration(disruptions_df, [t1, t2, t3])
                    except ValueError as e:
                        st.error(f"Invalid time format in thresholds: {e}")
                        disruptions_grouped = disruptions_df.copy()

                    st.markdown("""

                        The table below categorizes each disruption based on its duration using the specified thresholds.

                        - Helps quickly identify short vs. long outages.
                        - Use this to prioritize investigation or maintenance efforts.

                        """)
                    #pivoted_df = disruptions_pivot_summary(disruptions_grouped)
                    #st.dataframe(pivoted_df)
                    for group in disruptions_grouped['duration_group'].cat.categories:
                        st.markdown(f"##### Duration Group: {group}")

                        group_df = disruptions_grouped[disruptions_grouped['duration_group'] == group][
                            ['node_id', 'start', 'end', 'duration']
                        ].reset_index(drop=True)

                        # Convert timedelta to hh:mm:ss format
                        group_df['duration'] = group_df['duration'].apply(
                            lambda d: str(pd.to_timedelta(d)).split('.')[0]  # Remove microseconds
                        )

                        st.dataframe(group_df)
                    
                    




                    st.subheader("🌐 Network-Wide Disruptions")
                    st.markdown("""
                        This section identifies **time intervals where all sensor nodes experienced disruptions simultaneously**.  
                        To account for slight timing variations across nodes, a **time margin** is applied around each node's disruption period.

                        📏 **Disruption Margin Setting**  
                        Use the input box below to set the margin used to expand each node’s disruption window.  
                        This helps capture near-simultaneous outages across nodes.

                        Examples of valid input:
                        - `10min` (default)
                        - `30s` (30 seconds)
                        - `1h` (1 hour)

                        """)
                    

                    user_margin = st.text_input("Enter margin for overlap (e.g., 10min, 1h):", value="10min")
                    network_disruptions = find_network_disruptions(disruptions_df, margin=user_margin)
                    if not network_disruptions.empty:
                        st.markdown("### 📋 Detected Network-Wide Disruptions")

                        # Format each duration entry
                        df_disp = network_disruptions.copy()
                        df_disp['duration'] = df_disp['duration'].apply(
                            lambda d: str(pd.to_timedelta(d)).split('.')[0]  # Removes microseconds
                        )

                        st.dataframe(df_disp)

                        # Format total duration
                        total_duration = network_disruptions['duration'].sum()
                        formatted_total_duration = str(pd.to_timedelta(total_duration)).split('.')[0]

                        st.markdown(f"🕒 **Total Network-Wide Disruption Time:** `{formatted_total_duration}`")

                    else:
                        st.info("No network-wide disruptions were detected with the selected margin.")


                # Call the detection function
                
                if any(t in selected_table.lower() for t in ['scd', 'bme']):
                    st.markdown(f"### 🧪 Faulty Node Detection for `{selected_table}`")
                    st.markdown("""
                    This analysis identifies **sensor nodes that may be malfunctioning** based on unusual temperature patterns.  
                    A machine learning model detects **irregular readings**, and nodes are flagged as faulty if these irregularities persist for a significant duration.
                    """)

                    # Run detection
                    faulty_nodes, annotated_data = detect_faulty_nodes(df)
                    
                    if faulty_nodes:
                        st.markdown("### ⚠️ Faulty Nodes Detected")
                        st.markdown("""
                        The following nodes exhibited irregular temperature patterns over a sustained period.  
                        These may indicate sensor malfunction, environmental anomalies, or communication glitches:
                        """)
                        st.write(faulty_nodes)

                        st.markdown("### 📊 Faulty vs Total Readings per Node")
                        st.markdown("""
                        The bar chart below compares the **total number of readings** to the **number of readings flagged as faulty** for each node.  
                        Nodes with high counts of irregularities are strong candidates for inspection or maintenance.
                        """)
                        plot_faulty_readings_summary_streamlit(df, annotated_data)
                        # Statistical summary for all candidate features
                        features = candidate_features  # your list of features

                        faulty_data = annotated_data[annotated_data['predicted_irregularity']]
                        normal_data = annotated_data[~annotated_data['predicted_irregularity']]

                        st.markdown("### 📊 Statistical Summary of Faulty vs Non-Faulty Readings for All Features")
                        st.markdown(
                            "Below are descriptive statistics for features in faulty readings versus non-faulty readings. "
                            "This helps highlight deviations or anomalies across multiple sensor measurements."
                        )

                        # Calculate descriptive stats
                        faulty_stats = faulty_data[features].describe().round(2)
                        normal_stats = normal_data[features].describe().round(2)

                        # Rename index so both tables clearly label rows
                        faulty_stats.index = [f"{i}" for i in faulty_stats.index]
                        normal_stats.index = [f"{i}" for i in normal_stats.index]

                        # Display side by side using Streamlit columns
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("#### ✅ Normal Readings")
                            st.dataframe(normal_stats)

                        with col2:
                            st.markdown("#### ⚠️ Faulty Readings")
                            st.dataframe(faulty_stats)

                    else:
                        st.success("✅ No faulty nodes detected based on current model and thresholds.")
                        st.markdown("""
                        All sensor nodes appear to be operating within expected temperature ranges.  
                        No persistent irregularities were detected.
                        """)
                else:
                    st.info("ℹ️ Faulty node detection is only supported for sensor tables containing 'scd' or 'bme' in their names.")



                



                

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
        conn.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        if 'conn' in locals():
            conn.close() 
















def check_expected_vs_actual(df, start_date, end_date, interval_seconds=10, jitter_tolerance_seconds=10):
    """
    Compare expected vs actual timestamps for sensor node data.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least 'node_id' and 'timestamp' columns.
    start_date : str, datetime-like
        Start of the expected timestamp range. Can be a string or pandas.Timestamp.
    end_date : str, datetime-like
        End of the expected timestamp range. Can be a string or pandas.Timestamp.
    interval_seconds : int, optional
        Expected reporting interval in seconds. Default is 10.
    jitter_tolerance_seconds : int, optional
        Allowed jitter (tolerance) for nearest timestamp match in seconds. Default is 10.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing expected timestamps for each node, actual timestamps if found within tolerance,
        and a 'status' column indicating 'recorded' or 'missing'.
    """
    # Ensure datetime types
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    tolerance = pd.Timedelta(seconds=jitter_tolerance_seconds)
    expected_interval = pd.Timedelta(seconds=interval_seconds)

    expected_all = []
    nodes = df['node_id'].unique()

    for node in nodes:
        df_node = df[df['node_id'] == node].copy()
        if df_node.empty:
            continue

        start = min(df_node['timestamp'].min(), start_date)
        end = max(df_node['timestamp'].max(), end_date)

        expected_times = pd.date_range(start=start, end=end, freq=expected_interval)
        expected_df = pd.DataFrame({
            'node_id': node,
            'expected_time': expected_times
        })

        df_node = df_node[['timestamp']].sort_values('timestamp').drop_duplicates()
        df_node = df_node.rename(columns={'timestamp': 'actual_time'})

        merged = pd.merge_asof(expected_df.sort_values('expected_time'),
                               df_node.sort_values('actual_time'),
                               left_on='expected_time',
                               right_on='actual_time',
                               direction='nearest',
                               tolerance=tolerance)

        merged['status'] = np.where(merged['actual_time'].notna(), 'recorded', 'missing')
        expected_all.append(merged)

    merged_all = pd.concat(expected_all, ignore_index=True)
    merged_all['timestamp'] = merged_all['expected_time']
    return merged_all



def plot_recorded_vs_missing(results_df):
    """
    Plot recorded vs missing sensor readings over time for each node.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output from `check_expected_vs_actual()` function, must contain columns:
        'timestamp', 'node_id', and 'status' ('recorded' or 'missing').

    Returns
    -------
    None
        Displays the plot using Streamlit.
    """
    if results_df.empty:
        st.warning("No data available to plot.")
        return

    plt.figure(figsize=(15, 6))

    # Separate the two statuses
    recorded_df = results_df[results_df["status"] == "recorded"]
    missing_df = results_df[results_df["status"] == "missing"]

    # Plot recorded readings (green circles)
    sns.scatterplot(
        data=recorded_df,
        x="timestamp",
        y="node_id",
        color="green",
        marker='o',
        s=10,
        alpha=0.8,
        linewidth=0,
        label="recorded"
    )

    # Plot missing readings (red crosses)
    sns.scatterplot(
        data=missing_df,
        x="timestamp",
        y="node_id",
        color="#8B0000",
        marker='x',
        s=30,
        alpha=0.9,
        linewidth=1,
        label="missing"
    )

    plt.title("Recorded vs Missing Readings per Node Over Time", fontsize=14)
    plt.ylabel("Node")
    plt.xlabel("Timestamp")
    plt.legend(title="Status", loc="upper right")

    # Optional: Light gridlines for nodes
    for i, node in enumerate(sorted(results_df['node_id'].unique())):
        plt.axhline(y=node, color='gray', linewidth=0.2, alpha=0.3)

    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()


def calculate_uptime_summary(results_df):
    """
    Calculate uptime summary per node from expected vs actual results.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output from `check_expected_vs_actual()` function, must contain columns:
        'node_id' and 'status' (values: 'recorded' or 'missing').

    Returns
    -------
    pandas.DataFrame
        A DataFrame with recorded count, missing count, total, and uptime percentage per node.
    """
    if results_df.empty:
        return pd.DataFrame(columns=['recorded', 'missing', 'total', 'uptime_%'])

    summary = results_df.groupby(['node_id', 'status']).size().unstack(fill_value=0)
    summary['total'] = summary.sum(axis=1)
    summary['uptime_%'] = 100 * summary.get('recorded', 0) / summary['total']
    return summary


def plot_uptime_bar(uptime_summary_df):
    """
    Plot a bar chart of uptime percentage per node.

    Parameters
    ----------
    uptime_summary_df : pandas.DataFrame
        DataFrame with 'node_id' as index and a column 'uptime_%' representing uptime percentage.

    Returns
    -------
    None
        Displays the plot using Streamlit.
    """
    if uptime_summary_df.empty or 'uptime_%' not in uptime_summary_df.columns:
        st.warning("No uptime data available to plot.")
        return

    # Convert uptime_% to numeric if it's formatted as string
    uptime_summary_df['uptime_%'] = pd.to_numeric(uptime_summary_df['uptime_%'], errors='coerce')

    plt.figure(figsize=(8, 4))
    sns.barplot(
        data=uptime_summary_df.reset_index(),
        x="node_id",
        y="uptime_%",
        palette="viridis"
    )
    plt.title("Node Uptime Percentage")
    plt.ylabel("Uptime (%)")
    plt.xlabel("Node ID")
    plt.ylim(0, 100)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.close()



def find_disruptions(results_df, expected_interval=pd.Timedelta(seconds=10), min_duration='10min'):
    """
    Identify periods of consecutive missing data (disruptions) for each node.

    Parameters
    ----------
    results_df : pandas.DataFrame
        Output from `check_expected_vs_actual()` with columns 'node_id', 'timestamp', 'status'.
    expected_interval : pd.Timedelta
        Expected time interval between readings.
    min_duration : str or pd.Timedelta
        Minimum duration of a missing period to be considered a disruption.

    Returns
    -------
    pandas.DataFrame
        One row per disruption: node_id, start, end, duration.
    """
    disruptions = []
    nodes = results_df['node_id'].unique()
    min_duration = pd.Timedelta(min_duration)

    for node in nodes:
        node_data = results_df[results_df['node_id'] == node].sort_values("timestamp")
        node_data['gap'] = node_data['status'].ne(node_data['status'].shift()).cumsum()

        for _, group in node_data.groupby('gap'):
            if group['status'].iloc[0] == 'missing':
                duration = group['timestamp'].iloc[-1] - group['timestamp'].iloc[0] + expected_interval
                if duration >= min_duration:
                    disruptions.append({
                        "node_id": node,
                        "start": group['timestamp'].iloc[0],
                        "end": group['timestamp'].iloc[-1] + expected_interval,
                        "duration": duration
                    })

    return pd.DataFrame(disruptions)



def group_disruptions_by_duration(disruptions_df, thresholds=None):
    """
    Group disruptions into duration categories.

    Parameters
    ----------
    disruptions_df : pandas.DataFrame
        A DataFrame with a 'duration' column of type pd.Timedelta.
    thresholds : list of str or pd.Timedelta, optional
        A list of 3 thresholds defining the 4 duration buckets.
        Example: ['1D', '7D', '30D'] for <1 day, <1 week, <1 month, ≥1 month.

    Returns
    -------
    pandas.DataFrame
        The original DataFrame with an added 'duration_group' column.
    """

    if disruptions_df.empty or 'duration' not in disruptions_df.columns:
        return disruptions_df

    # Default thresholds
    if thresholds is None:
        thresholds = ['1D', '7D', '30D']
    thresholds = [pd.Timedelta(t) for t in thresholds]

    # Define bins and labels
    bins = [pd.Timedelta(0)] + thresholds + [pd.Timedelta.max]
    labels = [
        f"< {thresholds[0]}",
        f"{thresholds[0]} – {thresholds[1]}",
        f"{thresholds[1]} – {thresholds[2]}",
        f"≥ {thresholds[2]}"
    ]

    # Assign group
    disruptions_df = disruptions_df.copy()
    disruptions_df['duration_group'] = pd.cut(
        disruptions_df['duration'],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False
    )

    return disruptions_df


def disruptions_pivot_summary(disruptions_df):
    disruptions_df = disruptions_df.copy()
    disruptions_df['summary'] = (
        "Node: " + disruptions_df['node_id'].astype(str) +
        ", Start: " + disruptions_df['start'].astype(str) +
        ", End: " + disruptions_df['end'].astype(str)
    )
    pivot_df = disruptions_df.pivot_table(
        index=disruptions_df.index,
        columns='duration_group',
        values='summary',
        aggfunc='first'
    )
    return pivot_df.fillna('')




def find_network_disruptions(disruption_df, margin='10min'):
    """
    Identify time intervals where all nodes are simultaneously disrupted.

    Parameters:
    - disruption_df (DataFrame): Disruptions with columns ['node_id', 'start', 'end'].
    - margin (str or Timedelta): Padding applied before and after each disruption interval.
                                 Can be a string like '10min', '1h', etc.

    Returns:
    - DataFrame with columns ['start', 'end', 'duration'] for network-wide disruptions.
    """
    # Convert margin string to Timedelta if necessary
    if not isinstance(margin, pd.Timedelta):
        margin = pd.to_timedelta(margin)

    n_nodes = disruption_df['node_id'].nunique()

    # Expand each disruption interval by the margin
    expanded = disruption_df.copy()
    expanded['start_adj'] = expanded['start'] - margin
    expanded['end_adj'] = expanded['end'] + margin

    # Create all time points (start and end of each interval)
    time_points = sorted(set(expanded['start_adj']).union(set(expanded['end_adj'])))

    results = []
    for i in range(len(time_points) - 1):
        window_start = time_points[i]
        window_end = time_points[i + 1]

        # For this time window, check how many nodes are disrupted
        mask = (
            (expanded['start_adj'] <= window_start) &
            (expanded['end_adj'] >= window_end)
        )
        disrupted_nodes = expanded[mask]['node_id'].unique()

        if len(disrupted_nodes) == n_nodes:
            results.append({
                'start': window_start,
                'end': window_end,
                'duration': window_end - window_start
            })

    # Merge consecutive intervals
    if not results:
        return pd.DataFrame(columns=['start', 'end', 'duration'])

    merged = [results[0]]
    for entry in results[1:]:
        last = merged[-1]
        if entry['start'] <= last['end']:  # overlapping or adjacent
            merged[-1]['end'] = max(last['end'], entry['end'])
            merged[-1]['duration'] = merged[-1]['end'] - merged[-1]['start']
        else:
            merged.append(entry)

    return pd.DataFrame(merged)











def plot_network_disruptions(df):
    fig, ax = plt.subplots(figsize=(10, 2))

    for i, row in df.iterrows():
        ax.barh(y=1, width=row['duration'], left=row['start'], height=0.4, color='crimson', alpha=0.8)

    ax.set_yticks([])
    ax.set_title("⛔ Network-Wide Disruptions Timeline", fontsize=14)
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax.xaxis.set_tick_params(rotation=45)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig









def detect_faulty_nodes(new_data, model_path=os.path.join(os.path.dirname(__file__), 'irregularity_model.pkl'),
                        gap_threshold_sec=15, min_fault_duration_min=10):
    """
    Detects faulty nodes in temperature data based on a trained scikit-learn model.

    Parameters:
        new_data (pd.DataFrame): Data with 'timestamp', 'node_id', and 'temperature'.
        model_path (str): Path to the saved model (joblib format).
        gap_threshold_sec (int): Threshold in seconds to group contiguous irregularities.
        min_fault_duration_min (float): Minimum duration (in minutes) to classify a node as faulty.

    Returns:
        faulty_nodes (list): Node IDs identified as faulty.
        new_data (pd.DataFrame): Original data with added 'predicted_irregularity' column.
    """
    if new_data.empty:
        return [], new_data

    new_data = new_data.copy()
    new_data['timestamp'] = pd.to_datetime(new_data['timestamp'])
    new_data = new_data.sort_values(by=['node_id', 'timestamp'])

    # Load trained scikit-learn model
    model = joblib.load(model_path)

    # Scale temperature (in production, reuse training scaler)
    new_data = new_data.dropna(subset=['temperature'])

    scaler = joblib.load(os.path.join(os.path.dirname(__file__), 'scaler.pkl'))
    new_data['temp_scaled'] = scaler.transform(new_data[['temperature']])

    # Predict probability of irregularity and apply threshold
    X_new = new_data['temp_scaled'].values.reshape(-1, 1)
    probs = model.predict_proba(X_new)[:, 1]  # probability of class 1 (irregular)
    new_data['predicted_irregularity'] = (probs > 0.5).astype(bool)

    faulty_nodes = []

    for node_id, group in new_data[new_data['predicted_irregularity']].groupby('node_id'):
        group = group.sort_values('timestamp')
        group['time_diff'] = group['timestamp'].diff().dt.total_seconds().fillna(0)
        group['group'] = (group['time_diff'] > gap_threshold_sec).cumsum()

        durations = group.groupby('group')['timestamp'].agg(['min', 'max'])
        durations['duration_minutes'] = (durations['max'] - durations['min']).dt.total_seconds() / 60

        if any(durations['duration_minutes'] >= min_fault_duration_min):
            faulty_nodes.append(node_id)

    return faulty_nodes, new_data



def plot_faulty_readings_summary_streamlit(original_df, annotated_df):
    """
    Plots total vs. faulty temperature readings per node in Streamlit.

    Parameters:
        original_df (pd.DataFrame): Original dataset before annotation.
        annotated_df (pd.DataFrame): Data with 'predicted_irregularity' column.

    Returns:
        None (renders plot in Streamlit)
    """
    if 'predicted_irregularity' not in annotated_df.columns:
        st.error("annotated_df must contain 'predicted_irregularity' column.")
        return

    # Count total and faulty readings per node
    reading_counts = original_df.groupby('node_id').size().rename("total")
    faulty_counts = annotated_df[annotated_df['predicted_irregularity']].groupby('node_id').size().rename("faulty")

    # Merge for plotting
    summary_df = pd.concat([reading_counts, faulty_counts], axis=1).fillna(0).astype(int)

    # Bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    summary_df.plot(kind='bar', ax=ax, color=['skyblue', 'salmon'])
    ax.set_title("Total vs Faulty Readings per Node")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Reading Count")
    ax.legend(["Total Readings", "Faulty Readings"])
    plt.tight_layout()

    st.pyplot(fig)