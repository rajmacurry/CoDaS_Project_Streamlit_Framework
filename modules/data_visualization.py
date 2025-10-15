"""
Data Visualization Page - Interactive charts and graphs
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
    st.header("Exploratory Data Analysis")
    st.markdown("Explore and Visualize trends, patterns, and irregularites to better understand the data")
    
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

        # Date filtering for all tables (via measurements join)
        st.subheader("📅 Date Filtering")
        
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
        
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                key="db_end_date"
            )
        
        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date.")
            conn.close()
            return
        
        date_diff = (end_date - start_date).days
        if date_diff > 31:
            st.warning(f"⚠️ Date range is {date_diff} days. Consider a shorter range for better performance.")
        # Build query based on selected table
        if selected_table == 'measurements':
            query = f"""
            SELECT * FROM measurements 
            WHERE timestamp >= '{start_date}' 
            AND timestamp <= '{end_date} 23:59:59'
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
            WHERE m.timestamp >= '{start_date}'
            AND m.timestamp <= '{end_date} 23:59:59'
            ORDER BY m.timestamp DESC
            
            """
            #LIMIT 10000
        
        # Execute query and display results
        st.subheader(f"📊 {selected_table.title()} Data")
        
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
                        st.dataframe(df.head())
                
                st.subheader(f"📊 Statistics Summary for `{selected_table.title()}`")
                
                # To do the correlation of 10^-2 before describe of the data
                value = 0.01
                excluded_cols = {'timestamp', 'measurement_id', 'local_id', 'id', 'node_id'}
                candidate_features = [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]

                df[[f for f in candidate_features if f != "gas_resistance"]] = df[[f for f in candidate_features if f != "gas_resistance"]] * value
                st.dataframe(df.describe())


                # ----------------------------------------
                # Interquartile Analysis
                # ----------------------------------------

                st.subheader(f"📦 Interquartile Analysis")
                plot_interquartile_graph(df, candidate_features)


                plot_node_reading_counts(df, nodes=nodes)

                # --------------------------------------
                # 🧮 Correlation Adjustment Section
                # --------------------------------------

                

                st.subheader("🧮 Correlation Adjustment")
                st.markdown("Correlation already done for SCD30, BME680, and SPS30. ONLY REQUIED FOR NEW DATASET WITH DIFFERENT FEATURES")
                # Let user pick multiple features
                features_to_correct = st.multiselect("Select features to apply correction factors to", candidate_features)

                correction_factors = {}
                
                
                

                if features_to_correct:
                    for feature in features_to_correct:
                        factor = st.number_input(
                            f"Enter correction factor for '{feature}'",
                            value=1, format="%.5f", key=f"{feature}_factor"
                        )
                        correction_factors[feature] = factor

                    if st.button("✅ Apply Correction Factors and Continue"):
                        for feature, factor in correction_factors.items():
                            df[feature] = df[feature] * factor
                            st.success(f"Applied correction factor {factor} to '{feature}'")
                else:
                    st.info("Select at least one feature to apply corrections.")

                # --------------------------------------
                # 🚨 Outlier Detection & Correction
                # --------------------------------------

                
                st.subheader("🚨 Outlier Detection & Correction")
                with st.expander("✅ Start Outlier Detection", expanded=True):
                    method = st.radio(
                        "Choose outlier detection method:",
                        ("Use hardcoded bounds", "Use IQR filtering"),
                        index=0
                    )

                    use_iqr = method == "Use IQR filtering"
                    outlier_df, df_clean = check_and_remove_outliers(df, selected_table, use_iqr)


                    if not outlier_df.empty:
                        st.dataframe(outlier_df)

                        st.success(f"{len(outlier_df)} outlier rows detected and corrected.")

                        # Optional: Download CSV
                        csv = outlier_df.to_csv(index=False)
                        st.download_button("📥 Download Outliers as CSV", csv, file_name="outliers.csv")

                        # Optional: Plot per node
                        if 'node_id' in outlier_df.columns:
                            st.subheader("📊 Outlier Count per Node")
                            st.bar_chart(outlier_df['node_id'].value_counts().sort_index())

                        # Optional: Stats
                        st.subheader("📈 Outlier Stats")
                        st.dataframe(outlier_df.describe())
                        # Persist the corrected dataframe
                        df = df_clean
 
                    else:
                        st.info("No outliers detected.")
                    
                df = df_clean


                # --------------------------------------
                # 📊 Visualization Section
                # --------------------------------------
                st.markdown("### 📈 Feature Distributions")
                with st.expander("Distribution Analysis"):
                    plot_all_feature_distributions(df, candidate_features)


                st.subheader("📊 Feature Visualization")

                if candidate_features:
                    selected_feature = st.selectbox("Select a feature to visualize", candidate_features, key="plot_feature")

                    plot_line_by_node(data=df, y=selected_feature, title=f"{selected_feature.upper()} Over Time")


                    with st.expander("📊 Plots Per Node Id", expanded=False):
                        plot_line_by_each_node(df, y=selected_feature, title="Feature Over Time")


                    with st.expander("📊 Daily Feature Visualization", expanded=True):
                        if (date_diff+1) < 1:
                            st.error("Invalid date range.")
                        else:
                            max_days = min(14, date_diff+1)
                            default_days = min(3, max_days)
                            days_to_show = st.slider(
                                "Select number of days to visualize from the selected range",
                                min_value=1,
                                max_value=max_days,
                                value=default_days,
                                step=1,
                                key="visual_range_days"
                            )
                            # Let user select which N days (from within the start-end range)
                            available_dates = pd.date_range(start=start_date, end=end_date).to_pydatetime().tolist()
                            selected_day_start = st.selectbox(
                                "Pick start day for visualization",
                                available_dates[:(date_diff - days_to_show + 1)],
                                format_func=lambda x: x.strftime("%Y-%m-%d"),
                                key="vis_start_day"
                            )

                            vis_start = pd.to_datetime(selected_day_start).date()
                            vis_end = vis_start + pd.Timedelta(days=days_to_show - 1)

                            plot_hourly_feature_patterns(df, selected_feature, start_date=str(vis_start), end_date=str(vis_end))
                else:
                    st.warning("No numeric features available to visualize.")



                # --------------------------------------
                # Intrasensor Correlative Analysis
                # --------------------------------------



                
                


                # Call the correlation heatmap function
                plot_correlation_heatmap(df, candidate_features)

                st.subheader("📊 Correlative Feature Visualization")
                
                # --- Feature selection (max 2) ---
                if candidate_features:
                    selected_features = st.multiselect(
                        "Select two features to analyze",
                        candidate_features,
                        max_selections=2,
                        key="plot_features"
                    )

                # --- Proceed only if exactly 2 features are selected ---
                if len(selected_features) == 2:
                    # --- Date selection ---
                    min_date = df['timestamp'].dt.date.min()
                    max_date = df['timestamp'].dt.date.max()
                    default_date = min_date

                    selected_date = st.date_input(
                        "Select a start date to visualize",
                        value=default_date,
                        min_value=min_date,
                        max_value=max_date
                    )

                    # --- Number of days to visualize ---
                    num_days = st.slider("Number of days to show", 1, 7, value=2)

                    # --- Filter and plot ---
                    if selected_date:
                        correlativeAnalysis(df, selected_features, selected_date, num_days=num_days)


        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            
        conn.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        if 'conn' in locals():
            conn.close() 






def plot_line_by_node(data, y, title="Line Plot", hue="node_id", x="timestamp"):
    """
    Plots a line plot using mean-aggregated 1-minute resampled data in Streamlit.

    Parameters:
    - data: pd.DataFrame — input dataframe
    - y: str — y-axis variable
    - title: str — plot title
    - hue: str — column to color lines by (default: 'node_id')
    - x: str — timestamp column (default: 'timestamp')
    """

    # Ensure timestamp is datetime
    data[x] = pd.to_datetime(data[x])

    # Resample to 1-minute mean per node
    df_resampled = (
        data.set_index(x)
            .groupby(hue)[y]
            .resample("1min")
            .mean()
            .reset_index()
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.lineplot(data=df_resampled, x=x, y=y, hue=hue, palette='tab10', linewidth=1.2, ax=ax)

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel(x.capitalize(), fontsize=14)
    ax.set_ylabel(y.capitalize(), fontsize=14)
    plt.xticks(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # Show plot in Streamlit
    st.pyplot(fig)
    del df_resampled, data

def plot_line_by_each_node(data, y, title="Line Plot", x="timestamp"):
    """
    Plots individual line plots for each node_id using 1-minute mean-aggregated data.

    Parameters:
    - data: pd.DataFrame — input dataframe
    - y: str — y-axis variable to plot
    - title: str — base plot title
    - x: str — timestamp column (default: 'timestamp')
    """

    if x not in data.columns or y not in data.columns or 'node_id' not in data.columns:
        st.error("Data must include 'timestamp', 'node_id', and the selected feature.")
        return

    # Ensure timestamp is datetime
    data[x] = pd.to_datetime(data[x])

    # Resample: compute 1-min average per node
    resampled = (
        data.set_index(x)
            .groupby('node_id')[y]
            .resample("1min")
            .mean()
            .reset_index()
    )

    # Plot separately for each node_id
    for node in sorted(resampled['node_id'].unique()):
        node_df = resampled[resampled['node_id'] == node]

        fig, ax = plt.subplots(figsize=(14, 4))
        sns.lineplot(data=node_df, x=x, y=y, color='steelblue', ax=ax)

        ax.set_title(f"{title} — Node {node}", fontsize=14, fontweight='bold')
        ax.set_xlabel(x.capitalize())
        ax.set_ylabel(y.capitalize())
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)


def check_and_remove_outliers(df: pd.DataFrame, table_name: str, use_iqr: bool = False, iqr_factor: float = 0.0):
    """
    Detects and handles outliers in a sensor dataset using either hardcoded bounds or IQR-based filtering.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing sensor data.
    
    table_name : str
        Name of the sensor table, used to determine sensor type and corresponding bounds.
        Supports "scd", "bme", and "sps" sensor types (case-insensitive).

    use_iqr : bool, optional
        If True, apply IQR-based outlier filtering instead of using hardcoded bounds.
        For "sps" sensor data, IQR filtering is always applied regardless of this flag.

    iqr_factor : float, optional
        Scaling factor applied to the interquartile range (IQR) when correcting outliers.
        Outliers above the 90th percentile are capped at q90 + factor * IQR,
        and values below the 10th percentile are floored at q10 - factor * IQR.

    Returns:
    --------
    outlier_df : pd.DataFrame
        DataFrame containing rows identified as outliers (only for hardcoded bounds method).
        Returns an empty DataFrame if using IQR filtering.

    df_cleaned : pd.DataFrame
        Cleaned DataFrame with outliers removed (hardcoded bounds) or corrected (IQR filtering).

    Notes:
    ------
    - For "sps" sensor data, hardcoded bounds are not used due to high natural variance;
      IQR filtering is applied instead.
    - If the sensor type cannot be determined from `table_name`, no outlier handling is applied.
    """

    # Define bounds
    scd_bounds = {
        'co2': (100, 740),
        'temperature': (12, 32),
        'humidity': (14, 62.10)
    }

    bme_bounds = {
        'temperature_bme': (12, 32),
        'humidity_bme': (10.45, 61.45),
        'pressure': (61521, 103777),
        'gas_resistance': (3000, 5430598)
    }

    # Determine sensor type
    table_name_lower = table_name.lower()
    if "sps" in table_name_lower:
        st.warning("SPS data sensitive with high variance — applying IQR-based filtering.")
        use_iqr = True
        bounds = None
    elif "scd" in table_name_lower:
        bounds = scd_bounds
    elif "bme" in table_name_lower:
        bounds = bme_bounds
    else:
        st.warning("No matching sensor type for outlier removal.")
        return pd.DataFrame(), df

    # Apply IQR filtering if requested
    if use_iqr:
        st.info("Applying IQR-based *filtering* (removal) for outlier detection.")

        df_filtered = df.copy()
        numeric_cols = df_filtered.select_dtypes(include='number').columns

        # Create a mask for all rows that are within IQR bounds for all columns
        keep_mask = pd.Series(True, index=df_filtered.index)

        for col in numeric_cols:
            q1 = df_filtered[col].quantile(0.02)
            q3 = df_filtered[col].quantile(0.98)
            iqr_factor = 2.0
            iqr = q3 - q1
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr

            # Only keep values within the IQR range
            keep_mask &= df_filtered[col].between(lower_bound, upper_bound)

        outlier_df = df_filtered[~keep_mask]
        df_cleaned = df_filtered[keep_mask]

        return outlier_df, df_cleaned

    # Otherwise use hardcoded bounds
    df_cleaned = df.copy()
    outlier_mask = pd.Series(False, index=df.index)

    for col, (low, high) in bounds.items():
        if col in df.columns:
            col_outliers = (df[col] < low) | (df[col] > high)
            outlier_mask |= col_outliers

    outlier_df = df[outlier_mask].copy()
    df_cleaned = df[~outlier_mask].copy()

    return outlier_df, df_cleaned


def plot_hourly_feature_patterns(
    df: pd.DataFrame,
    selected_feature: str,
    start_date: str = None,
    end_date: str = None,
    last_n_days: int = None
):
    """
    Plot hourly average of a selected feature grouped by node and day.

    Args:
        df (pd.DataFrame): The input DataFrame with 'timestamp', 'node_id', and feature columns.
        selected_feature (str): The numeric feature to visualize.
        start_date (str, optional): Start date in 'YYYY-MM-DD' format.
        end_date (str, optional): End date in 'YYYY-MM-DD' format.
        last_n_days (int, optional): If provided, overrides start/end and plots last N days.
    """
    if 'timestamp' not in df.columns or 'node_id' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' and 'node_id' columns.")

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Filter by date
    if last_n_days is not None:
        max_date = df['date'].max()
        min_date = max_date - pd.Timedelta(days=last_n_days - 1)
        df = df[df['date'].between(min_date, max_date)]
    elif start_date and end_date:
        df = df[df['date'].between(pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date())]

    if df.empty:
        st.warning("No data available in the selected date range.")
        return

    # Compute hourly average per node per day
    hourly_df = (
        df.groupby(['date', 'hour', 'node_id'])[selected_feature]
        .mean()
        .reset_index()
    )

    # Plot
    g = sns.FacetGrid(hourly_df, row='date', height=2.5, aspect=2, sharey=False)
    g.map_dataframe(sns.lineplot, x='hour', y=selected_feature, hue='node_id', palette='tab10')
    g.add_legend(title='Node ID')

    g.set_titles(col_template="{col_name}")
    g.set_axis_labels("Hour", f"Avg {selected_feature.upper()}")
    for ax in g.axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Daily Hourly Patterns by Node for '{selected_feature}'", fontsize=16, fontweight='bold')
    st.pyplot(g.fig)




def plot_all_feature_distributions(
        data, candidate_features
    ):
    """
    Plots the feature distribution

    Args: 
        df (pd.DataFrame), similary to previous plot_hourly_feature_patterns.
        candidate_features, the actual features computed in the show() function

    """


    for feature in candidate_features:
        if feature not in data.columns:
            continue  # Skip if feature doesn't exist

        feature_data = data[feature].dropna()

        if feature_data.empty:
            continue  # Skip empty columns

        st.markdown(f"#### 📊 {feature} Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(feature_data, bins=10, kde=True, color="#C76D46", ax=ax)

        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
        ax.set_title(f"{feature} Distribution")

        st.pyplot(fig)


def check_and_remove_outliers2(df: pd.DataFrame, table_name: str):
    # Define bounds
    # Define bounds
    scd_bounds = {
        'co2': (100, 740),
        'temperature': (12, 32),
        'humidity': (14, 62.10)
    }

    bme_bounds = {
        'temperature_bme': (12, 32),
        'humidity_bme': (10.45, 61.45),
        'pressure': (61521, 103777),
        'gas_resistance': (3000, 5430598)
    }

    # Check table type
    table_name_lower = table_name.lower()
    
    if "sps" in table_name_lower:
        st.warning("SPS data sensitive with high variance — applying IQR-based filtering instead of removal.")

        df_filtered = df.copy()
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                q10 = df[col].quantile(0.10)
                q90 = df[col].quantile(0.90)
                iqr = q90 - q10
                iqr_factor = 0.1
                upper_replacement = q90 + iqr_factor * iqr
                lower_replacement = q10 - iqr_factor * iqr

                df_filtered[col] = df[col].apply(
                    lambda x: upper_replacement if x > q90 else (lower_replacement if x < q10 else x)
                )

        # Return empty outlier_df since no records are dropped, just corrected
        return pd.DataFrame(), df_filtered

    elif "scd" in table_name_lower:
        bounds = scd_bounds
    elif "bme" in table_name_lower:
        bounds = bme_bounds
    else:
        st.warning("No matching sensor type for outlier removal.")
        return pd.DataFrame(), df

    df_cleaned = df.copy()
    outlier_mask = pd.Series(False, index=df.index)

    for col, (low, high) in bounds.items():
        if col in df.columns:
            col_outliers = (df[col] < low) | (df[col] > high)
            outlier_mask |= col_outliers

    outlier_df = df[outlier_mask].copy()
    df_cleaned = df[~outlier_mask].copy()

    return outlier_df, df_cleaned




def correlativeAnalysis(
        df, features, start_date, num_days
    ):

    """
    Correlative Analysis for features from the same sensor.

    Args:
    Dataframe
    features, used in the analysis
    start_date,
    Number of days
    """

    feature1, feature2 = features

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date

    # Generate date range
    end_date = start_date + pd.Timedelta(days=num_days - 1)
    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
    filtered = df[mask].copy()

    if filtered.empty:
        st.warning(f"No data available between {start_date} and {end_date}.")
        return

    # Melt to long-form
    long_df = pd.melt(
        filtered,
        id_vars=['timestamp', 'hour', 'node_id', 'date'],
        value_vars=[feature1, feature2],
        var_name='feature',
        value_name='value'
    )

    # Group by date, hour, node, and feature
    hourly_avg = (
        long_df
        .groupby(['date', 'feature', 'hour', 'node_id'])['value']
        .mean()
        .reset_index()
    )

    # Create FacetGrid: columns = features, rows = dates
    g = sns.FacetGrid(
        hourly_avg,
        row='date',
        col='feature',
        hue='node_id',
        height=3.5,
        aspect=1.6,
        sharey=False
    )

    g.map(sns.lineplot, 'hour', 'value')
    g.add_legend(title='Node ID')
    g.set_axis_labels("Hour", "Avg Value")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")

    for ax in g.axes.flatten():
        ax.grid(True, linestyle='--', alpha=0.5)

    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(f"Hourly Averages per Feature per Day ({start_date} to {end_date})", fontsize=16, fontweight='bold')
    st.pyplot(g.fig)


def plot_node_reading_counts(df: pd.DataFrame, nodes: list):
    """
    Plot the number of readings for each node in the provided list using Streamlit.
    Includes nodes with 0 readings.
    """

    if 'node_id' not in df.columns:
        st.error("❌ 'node_id' column not found in the dataset.")
        return

    # Count readings per node
    actual_counts = df['node_id'].value_counts()

    # Fill in zero for missing nodes
    counts = pd.Series({node: actual_counts.get(node, 0) for node in nodes}).sort_index()

    # Display table
    st.subheader("📊 Count per Node")
    st.dataframe(counts.rename("reading_count").rename_axis("node_id").reset_index())

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, palette="Set2", ax=ax)
    ax.set_title("Reading Counts per Node ID")
    ax.set_xlabel("Node ID")
    ax.set_ylabel("Reading Count")
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    st.pyplot(fig)


def plot_correlation_heatmap(df: pd.DataFrame, candidate_features: list):
    """
    Display a correlation heatmap for the selected numeric features.

    Args:
        df (pd.DataFrame): The input DataFrame containing numeric data.
        candidate_features (list): List of feature names to include in the heatmap.
    """
    if not candidate_features:
        st.warning("No candidate features provided for correlation analysis.")
        return

    # Filter numeric features present in both df and candidate_features
    features = [f for f in candidate_features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    
    if len(features) < 2:
        st.warning("At least two numeric features are required to compute correlations.")
        return

    corr_matrix = df[features].corr()

    st.subheader("🔗 Feature Correlation Heatmap")
    st.markdown(
                    "This heatmap shows the pairwise Pearson correlation coefficients between selected sensor features. "
                    "Use it to identify strongly related variables (positive or negative)."
                )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75}
    )
    plt.title("Correlation Matrix", fontsize=14, fontweight='bold')
    st.pyplot(plt.gcf())


def plot_interquartile_graph(df: pd.DataFrame, features: list):
    """
    Plot individual horizontal box plots showing interquartile ranges (IQR) for each selected feature.

    Args:
        df (pd.DataFrame): The input DataFrame with numeric features.
        features (list): List of feature names to include in the plot.
    """
    st.markdown("### Interquartile Range (IQR) for Each Feature")

    for feature in features:
        if feature not in df.columns:
            continue  # Skip missing features

        st.markdown(f"**Feature:** `{feature}`")

        plt.figure(figsize=(8, 1.5))
        sns.boxplot(x=df[feature], orient='h', color='skyblue')
        plt.xlabel("Value")
        plt.title(f"IQR of {feature}", fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        st.pyplot(plt.gcf())
        plt.close()