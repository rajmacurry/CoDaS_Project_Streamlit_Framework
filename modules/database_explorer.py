"""
Database Explorer Page - Browse and explore database tables
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# Known table schema
TABLES_SCHEMA = {
    "measurements": {
        "columns": ["id", "node_id", "timestamp"],
        "datetime_col": "timestamp",
        "description": "Main measurements table with timestamps",
    },
    "sensor_bme680": {
        "columns": [
            "id",
            "measurement_id",
            "humidity",
            "temperature",
            "gas_resistance",
            "pressure",
        ],
        "join_col": "measurement_id",
        "description": "BME680 sensor data (humidity, temperature, gas, pressure)",
    },
    "sensor_scd30": {
        "columns": ["id", "measurement_id", "co2", "temperature", "humidity"],
        "join_col": "measurement_id",
        "description": "SCD30 sensor data (CO2, temperature, humidity)",
    },
    "sensor_sps30": {
        "columns": [
            "id",
            "measurement_id",
            "mass_conc_pm1",
            "mass_conc_pm2_5",
            "mass_conc_pm4",
            "mass_conc_pm10",
            "num_conc_pm0_5",
            "num_conc_pm1",
            "num_conc_pm2_5",
            "num_conc_pm4",
            "num_conc_pm10",
            "particle_size",
        ],
        "join_col": "measurement_id",
        "description": "SPS30 sensor data (particulate matter concentrations)",
    },
}


def get_date_range_for_table(conn, default_days=30):
    """Get appropriate date range for measurements table"""
    try:
        # Get data range from measurements table
        date_query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM measurements"
        date_df = pd.read_sql_query(date_query, conn)

        if not date_df.empty and date_df["min_date"].iloc[0]:
            min_date = pd.to_datetime(date_df["min_date"].iloc[0]).date()
            max_date = pd.to_datetime(date_df["max_date"].iloc[0]).date()

            # Default to last month of available data
            default_start = max_date - timedelta(days=default_days)
            if default_start < min_date:
                default_start = min_date

            return min_date, max_date, default_start, max_date
        else:
            # Fallback for empty data - use 2021-2022 range
            return (
                datetime(2021, 1, 1).date(),
                datetime(2022, 12, 31).date(),
                datetime(2022, 12, 1).date(),
                datetime(2022, 12, 31).date(),
            )
    except:
        # Fallback dates for 2021-2022
        return (
            datetime(2021, 1, 1).date(),
            datetime(2022, 12, 31).date(),
            datetime(2022, 12, 1).date(),
            datetime(2022, 12, 31).date(),
        )


def show():
    st.header("🗃️ Database Explorer")
    st.markdown(" Browse and explore your sensor database tables with date filtering.")

    db_path = os.getenv("DATABASE_PATH", "sensor_data.db")

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

        # Display schema overview
        with st.expander("📋 Database Schema Overview", expanded=False):
            st.markdown("**Sensor Database Structure:**")
            for table_name, schema in TABLES_SCHEMA.items():
                if table_name in available_tables:
                    st.markdown(f"**{table_name}:** {schema['description']}")
                    st.markdown(f"Columns: `{', '.join(schema['columns'])}`")
                    if "datetime_col" in schema:
                        st.markdown(f"🕒 Timestamp column: `{schema['datetime_col']}`")
                    elif "join_col" in schema:
                        st.markdown(
                            f"🔗 Links to measurements via: `{schema['join_col']}`"
                        )
                    st.markdown("---")

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

        # Initialize session state for date values if not present
        if "selected_start_date" not in st.session_state:
            st.session_state.selected_start_date = default_start
        if "selected_end_date" not in st.session_state:
            st.session_state.selected_end_date = default_end

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date:",
                value=st.session_state.selected_start_date,
                min_value=min_date,
                max_value=max_date,
                key="db_start_date",
            )

        with col2:
            end_date = st.date_input(
                "End Date:",
                value=st.session_state.selected_end_date,
                min_value=min_date,
                max_value=max_date,
                key="db_end_date",
            )

        # Update session state when widgets change
        if start_date != st.session_state.selected_start_date:
            st.session_state.selected_start_date = start_date
        if end_date != st.session_state.selected_end_date:
            st.session_state.selected_end_date = end_date

        # Use session state values for consistency (especially after quick select buttons)
        query_start_date = st.session_state.selected_start_date
        query_end_date = st.session_state.selected_end_date

        # Validate date range
        if query_start_date > query_end_date:
            st.error("Start date must be before end date.")
            conn.close()
            return

        date_diff = (query_end_date - query_start_date).days
        if date_diff > 31:
            st.warning(
                f"⚠️ Date range is {date_diff} days. Consider a shorter range for better performance."
            )

        # Time period quick select
        st.markdown("**Quick Select:**")

        # Generate dynamic date options based on actual data
        data_start_year = min_date.year
        data_end_year = max_date.year

        # Create columns based on how many options we have
        quick_options = []
        quick_options.append(("Last 7 Days", max_date - timedelta(days=7), max_date))
        quick_options.append(("Last Month", max_date - timedelta(days=30), max_date))

        # Add the most recent complete month if we have data
        if max_date.day > 1:  # Not the first day of month
            prev_month = max_date.replace(day=1) - timedelta(days=1)
            month_start = prev_month.replace(day=1)
            quick_options.append(
                (f"{prev_month.strftime('%b %Y')}", month_start, prev_month)
            )

        # Add full data range option
        quick_options.append(("All Data", min_date, max_date))

        # Create columns for buttons
        cols = st.columns(len(quick_options))

        for i, (label, start_date_opt, end_date_opt) in enumerate(quick_options):
            with cols[i]:
                if st.button(label):
                    st.session_state.selected_start_date = max(start_date_opt, min_date)
                    st.session_state.selected_end_date = min(end_date_opt, max_date)
                    st.rerun()

        # Build query based on selected table
        if selected_table == "measurements":
            query = f"""
            SELECT * FROM measurements 
            WHERE timestamp >= '{query_start_date}' 
            AND timestamp <= '{query_end_date} 23:59:59'
            ORDER BY timestamp DESC
            LIMIT 10000
            """
        else:
            # Join with measurements table for date filtering
            schema = TABLES_SCHEMA[selected_table]
            join_col = schema["join_col"]

            query = f"""
            SELECT s.*, m.timestamp, m.node_id
            FROM {selected_table} s
            JOIN measurements m ON s.{join_col} = m.id
            WHERE m.timestamp >= '{query_start_date}'
            AND m.timestamp <= '{query_end_date} 23:59:59'
            ORDER BY m.timestamp DESC
            LIMIT 10000
            """

        # Execute query and display results
        st.subheader(f"📊 {selected_table.title()} Data")

        # Show current date range being queried
        st.info(
            f"📅 Querying data from **{query_start_date}** to **{query_end_date}** ({date_diff} days)"
        )

        try:
            with st.spinner("Loading data..."):
                df = pd.read_sql_query(query, conn)

            if df.empty:
                st.warning(
                    f"No data found in {selected_table} for the selected date range."
                )
            else:
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Records Found", f"{len(df):,}")
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    if date_diff is not None:
                        st.metric("Date Range", f"{date_diff} days")
                    elif "timestamp" in df.columns:
                        date_range_days = (
                            pd.to_datetime(df["timestamp"].max())
                            - pd.to_datetime(df["timestamp"].min())
                        ).days
                        st.metric("Date Range", f"{date_range_days} days")
                    else:
                        st.metric("Date Range", f"{date_diff} days")
                with col4:
                    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                    st.metric("Memory", f"{memory_mb:.2f} MB")

                # Display data with option to show all or preview
                show_all = st.checkbox("Show all data", value=len(df) <= 1000)

                if show_all or len(df) <= 1000:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"Showing first 1000 of {len(df):,} records")
                    st.dataframe(df.head(1000), use_container_width=True)

                # Quick data insights
                with st.expander("📈 Quick Data Insights"):
                    # Numeric columns summary
                    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
                    if numeric_cols:
                        st.write("**Numeric Columns Summary:**")
                        st.dataframe(
                            df[numeric_cols].describe(), use_container_width=True
                        )

                    # Show unique values for categorical columns
                    categorical_cols = df.select_dtypes(
                        include=["object"]
                    ).columns.tolist()
                    if categorical_cols:
                        st.write("**Categorical Columns:**")
                        for col in categorical_cols:
                            unique_count = df[col].nunique()
                            st.write(f"- **{col}**: {unique_count} unique values")

                    # Time series info if timestamp available
                    if "timestamp" in df.columns:
                        st.write("**Time Series Info:**")
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        st.write(
                            f"- **Time Range**: {df['timestamp'].min()} to {df['timestamp'].max()}"
                        )
                        st.write(
                            f"- **Total Duration**: {(df['timestamp'].max() - df['timestamp'].min()).days} days"
                        )
                        st.write(
                            f"- **Average Interval**: {df['timestamp'].diff().mean()}"
                        )

                # Download option
                if st.button("📥 Download CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name=f"{selected_table}_{start_date}_{end_date}.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

        conn.close()

    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        if "conn" in locals():
            conn.close()
