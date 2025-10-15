"""
Table Statistics Page - Display comprehensive statistics for database tables
"""

import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

# Known table schema
TABLES_SCHEMA = {
    'measurements': {
        'columns': ['id', 'node_id', 'timestamp'],
        'datetime_col': 'timestamp',
        'description': 'Main measurements table with timestamps',
        'key_metrics': ['id', 'node_id']
    },
    'sensor_bme680': {
        'columns': ['id', 'measurement_id', 'humidity', 'temperature', 'gas_resistance', 'pressure'],
        'join_col': 'measurement_id',
        'description': 'BME680 sensor data (humidity, temperature, gas, pressure)',
        'key_metrics': ['humidity', 'temperature', 'gas_resistance', 'pressure']
    },
    'sensor_scd30': {
        'columns': ['id', 'measurement_id', 'co2', 'temperature', 'humidity'],
        'join_col': 'measurement_id',
        'description': 'SCD30 sensor data (CO2, temperature, humidity)',
        'key_metrics': ['co2', 'temperature', 'humidity']
    },
    'sensor_sps30': {
        'columns': ['id', 'measurement_id', 'mass_conc_pm1', 'mass_conc_pm2_5', 'mass_conc_pm4', 'mass_conc_pm10', 'num_conc_pm0_5', 'num_conc_pm1', 'num_conc_pm2_5', 'num_conc_pm4', 'num_conc_pm10', 'particle_size'],
        'join_col': 'measurement_id',
        'description': 'SPS30 sensor data (particulate matter concentrations)',
        'key_metrics': ['mass_conc_pm2_5', 'mass_conc_pm10', 'num_conc_pm2_5', 'particle_size']
    }
}

def get_date_range_for_table(conn, default_days=7):
    """Get appropriate date range for measurements table"""
    try:
        date_query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM measurements"
        date_df = pd.read_sql_query(date_query, conn)
        
        if not date_df.empty and date_df['min_date'].iloc[0]:
            min_date = pd.to_datetime(date_df['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(date_df['max_date'].iloc[0]).date()
            
            # Default to last week of available data
            default_start = max_date - timedelta(days=default_days)
            if default_start < min_date:
                default_start = min_date
                
            return min_date, max_date, default_start, max_date
        else:
            # Fallback for empty data
            return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date(), datetime(2022, 12, 24).date(), datetime(2022, 12, 31).date()
    except:
        return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date(), datetime(2022, 12, 24).date(), datetime(2022, 12, 31).date()

def calculate_statistics(df, table_name):
    """Calculate comprehensive statistics for a dataframe"""
    stats = {}
    
    # Basic info
    stats['total_records'] = len(df)
    stats['total_columns'] = len(df.columns)
    stats['memory_usage'] = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        stats['numeric_summary'] = df[numeric_cols].describe()
        stats['missing_values'] = df[numeric_cols].isnull().sum()
        stats['data_types'] = df[numeric_cols].dtypes
    
    # Time series analysis if timestamp available
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        stats['time_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'avg_interval': df['timestamp'].diff().mean()
        }
    
    # Node analysis for multi-node data
    if 'node_id' in df.columns:
        stats['node_analysis'] = {
            'unique_nodes': df['node_id'].nunique(),
            'records_per_node': df['node_id'].value_counts(),
            'nodes_list': sorted(df['node_id'].unique())
        }
    
    return stats

def show():
    st.header("📊 Table Statistics")
    st.markdown("Comprehensive statistical analysis of your sensor database tables.")
    
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
        
        table_options = [t for t in TABLES_SCHEMA.keys() if t in available_tables]
        
        if not table_options:
            st.error("Expected sensor tables not found in database.")
            conn.close()
            return
        
        # Table selection
        st.subheader("🎯 Table Selection")
        selected_table = st.selectbox("Choose table for statistics:", table_options)
        
        # Date filtering
        st.subheader("📅 Date Filtering")
        
        min_date, max_date, default_start, default_end = get_date_range_for_table(conn)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=default_start,
                min_value=min_date,
                max_value=max_date,
                key="stats_start_date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date:",
                value=default_end,
                min_value=min_date,
                max_value=max_date,
                key="stats_end_date"
            )
        
        with col3:
            # Generate dynamic time period options based on actual data
            data_start_year = min_date.year
            data_end_year = max_date.year
            
            # Create dynamic options
            time_period_options = ["Custom", "Last 7 Days", "Last Month"]
            
            # Add the most recent complete month if we have data
            if max_date.day > 1:  # Not the first day of month
                prev_month = max_date.replace(day=1) - timedelta(days=1)
                time_period_options.append(f"{prev_month.strftime('%b %Y')}")
            
            # Add yearly options for available years
            for year in range(data_end_year, data_start_year - 1, -1):
                time_period_options.append(f"All {year}")
            
            # Add full data range option
            time_period_options.append("All Data")
            
            time_period = st.selectbox(
                "Quick Select:",
                time_period_options,
                key="stats_period"
            )
        
        # Handle quick select
        if time_period != "Custom":
            if time_period == "Last 7 Days":
                start_date = max_date - timedelta(days=7)
                end_date = max_date
            elif time_period == "Last Month":
                start_date = max_date - timedelta(days=30)
                end_date = max_date
            elif time_period == "All Data":
                start_date = min_date
                end_date = max_date
            elif time_period.startswith("All "):
                # Handle "All YYYY" format
                year = int(time_period.split()[1])
                start_date = datetime(year, 1, 1).date()
                end_date = datetime(year, 12, 31).date()
                # Ensure dates are within actual data range
                start_date = max(start_date, min_date)
                end_date = min(end_date, max_date)
            elif len(time_period.split()) == 2:
                # Handle "Mon YYYY" format (e.g., "Dec 2022")
                try:
                    month_year = datetime.strptime(time_period, "%b %Y").date()
                    start_date = month_year.replace(day=1)
                    # Get last day of month
                    if month_year.month == 12:
                        end_date = datetime(month_year.year + 1, 1, 1).date() - timedelta(days=1)
                    else:
                        end_date = datetime(month_year.year, month_year.month + 1, 1).date() - timedelta(days=1)
                    # Ensure dates are within actual data range
                    start_date = max(start_date, min_date)
                    end_date = min(end_date, max_date)
                except ValueError:
                    # Fallback to current selection if parsing fails
                    pass
        
        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date.")
            conn.close()
            return
        
        date_diff = (end_date - start_date).days
        if date_diff > 31:
            st.warning(f"⚠️ Date range is {date_diff} days. Large ranges may take longer to process.")
        
        # Build query
        if selected_table == 'measurements':
            query = f"""
            SELECT * FROM measurements 
            WHERE timestamp >= '{start_date}' 
            AND timestamp <= '{end_date} 23:59:59'
            ORDER BY timestamp DESC
            LIMIT 50000
            """
        else:
            # Join with measurements for date filtering
            schema = TABLES_SCHEMA[selected_table]
            join_col = schema['join_col']
            
            query = f"""
            SELECT s.*, m.timestamp, m.node_id
            FROM {selected_table} s
            JOIN measurements m ON s.{join_col} = m.id
            WHERE m.timestamp >= '{start_date}'
            AND m.timestamp <= '{end_date} 23:59:59'
            ORDER BY m.timestamp DESC
            LIMIT 50000
            """
        
        # Execute query and calculate statistics
        try:
            with st.spinner("Loading and analyzing data..."):
                df = pd.read_sql_query(query, conn)
            
            if df.empty:
                st.warning(f"No data found in {selected_table} for the selected date range.")
                conn.close()
                return
            
            # Calculate comprehensive statistics
            stats = calculate_statistics(df, selected_table)
            
            # Display overview metrics
            st.subheader(f"📈 {selected_table.title()} Statistics Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{stats['total_records']:,}")
            with col2:
                st.metric("Columns", stats['total_columns'])
            with col3:
                st.metric("Memory Usage", f"{stats['memory_usage']:.2f} MB")
            with col4:
                if 'time_range' in stats:
                    st.metric("Duration", f"{stats['time_range']['duration_days']} days")
                else:
                    st.metric("Date Range", f"{date_diff} days")
            
            # Time series statistics
            if 'time_range' in stats:
                st.subheader("⏰ Time Series Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Time Range:**")
                    st.write(f"Start: {stats['time_range']['start']}")
                    st.write(f"End: {stats['time_range']['end']}")
                    st.write(f"Duration: {stats['time_range']['duration_days']} days")
                
                with col2:
                    st.write("**Data Frequency:**")
                    avg_interval = stats['time_range']['avg_interval']
                    if pd.notna(avg_interval):
                        st.write(f"Avg Interval: {avg_interval}")
                        records_per_day = stats['total_records'] / max(1, stats['time_range']['duration_days'])
                        st.write(f"Records/Day: {records_per_day:.1f}")
            
            # Node analysis
            if 'node_analysis' in stats:
                st.subheader("🔗 Multi-Node Analysis")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Unique Nodes", stats['node_analysis']['unique_nodes'])
                    st.write("**Node IDs:**", ", ".join(map(str, stats['node_analysis']['nodes_list'])))
                
                with col2:
                    st.write("**Records per Node:**")
                    node_counts = stats['node_analysis']['records_per_node']
                    for node, count in node_counts.head(10).items():
                        st.write(f"Node {node}: {count:,} records")
            
            # Numeric statistics
            if 'numeric_summary' in stats:
                st.subheader("🔢 Numeric Data Analysis")
                
                # Key metrics based on table type
                schema = TABLES_SCHEMA[selected_table]
                key_metrics = schema.get('key_metrics', [])
                
                if key_metrics:
                    available_metrics = [col for col in key_metrics if col in df.columns]
                    if available_metrics:
                        st.write(f"**Key Metrics for {selected_table}:**")
                        key_stats = df[available_metrics].describe()
                        st.dataframe(key_stats, use_container_width=True)
                
                # Full numeric summary
                with st.expander("📋 Complete Numeric Summary"):
                    st.dataframe(stats['numeric_summary'], use_container_width=True)
                
                # Missing values analysis
                if stats['missing_values'].sum() > 0:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame({
                        'Column': stats['missing_values'].index,
                        'Missing Count': stats['missing_values'].values,
                        'Missing %': (stats['missing_values'].values / len(df) * 100).round(2)
                    })
                    st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            # Data visualizations
            st.subheader("📊 Data Visualizations")
            
            # Time series plot if available
            if 'timestamp' in df.columns and len(df) > 1:
                schema = TABLES_SCHEMA[selected_table]
                key_metrics = schema.get('key_metrics', [])
                available_metrics = [col for col in key_metrics if col in df.columns and df[col].dtype in ['float64', 'int64']]
                
                if available_metrics:
                    metric_to_plot = st.selectbox("Select metric to plot over time:", available_metrics)
                    
                    if metric_to_plot:
                        # Create time series plot
                        fig = px.line(
                            df.sort_values('timestamp'), 
                            x='timestamp', 
                            y=metric_to_plot,
                            title=f"{metric_to_plot.title()} Over Time",
                            color='node_id' if 'node_id' in df.columns and df['node_id'].nunique() > 1 else None
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
            
            # Distribution plots
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col1, col2 = st.columns(2)
                
                with col1:
                    if len(numeric_cols) > 0:
                        hist_col = st.selectbox("Histogram:", numeric_cols, key="hist_col")
                        if hist_col:
                            fig = px.histogram(df, x=hist_col, title=f"Distribution of {hist_col.title()}")
                            st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    if len(numeric_cols) > 1:
                        scatter_x = st.selectbox("Scatter X:", numeric_cols, key="scatter_x")
                        scatter_y = st.selectbox("Scatter Y:", [col for col in numeric_cols if col != scatter_x], key="scatter_y")
                        
                        if scatter_x and scatter_y:
                            fig = px.scatter(
                                df, x=scatter_x, y=scatter_y, 
                                title=f"{scatter_y.title()} vs {scatter_x.title()}",
                                color='node_id' if 'node_id' in df.columns and df['node_id'].nunique() > 1 else None
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            # Export options
            st.subheader("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Statistics Summary"):
                    summary_data = {
                        'Table': selected_table,
                        'Date Range': f"{start_date} to {end_date}",
                        'Total Records': stats['total_records'],
                        'Columns': stats['total_columns'],
                        'Memory Usage (MB)': stats['memory_usage']
                    }
                    
                    if 'time_range' in stats:
                        summary_data.update({
                            'Duration (days)': stats['time_range']['duration_days'],
                            'Start Time': stats['time_range']['start'],
                            'End Time': stats['time_range']['end']
                        })
                    
                    summary_df = pd.DataFrame([summary_data])
                    csv = summary_df.to_csv(index=False)
                    st.download_button(
                        "Click to Download Summary",
                        csv,
                        f"{selected_table}_statistics_{start_date}_{end_date}.csv",
                        "text/csv"
                    )
            
            with col2:
                if st.button("Download Raw Data Sample"):
                    sample_size = min(10000, len(df))
                    sample_df = df.head(sample_size)
                    csv = sample_df.to_csv(index=False)
                    st.download_button(
                        "Click to Download Sample",
                        csv,
                        f"{selected_table}_sample_{start_date}_{end_date}.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error analyzing data: {str(e)}")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        if 'conn' in locals():
            conn.close() 