"""
Query Interface Page - Execute custom SQL queries
"""

import streamlit as st
import sqlite3
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# Known sensor database schema
SENSOR_TABLES_SCHEMA = {
    'measurements': {
        'columns': ['id', 'node_id', 'timestamp'],
        'datetime_col': 'timestamp',
        'description': 'Main measurements table with timestamps and node identifiers'
    },
    'sensor_bme680': {
        'columns': ['id', 'measurement_id', 'humidity', 'temperature', 'gas_resistance', 'pressure'],
        'join_col': 'measurement_id',
        'description': 'BME680 environmental sensor (humidity, temperature, gas, pressure)'
    },
    'sensor_scd30': {
        'columns': ['id', 'measurement_id', 'co2', 'temperature', 'humidity'],
        'join_col': 'measurement_id',
        'description': 'SCD30 CO2 sensor (carbon dioxide, temperature, humidity)'
    },
    'sensor_sps30': {
        'columns': ['id', 'measurement_id', 'mass_conc_pm1', 'mass_conc_pm2_5', 'mass_conc_pm4', 'mass_conc_pm10', 'num_conc_pm0_5', 'num_conc_pm1', 'num_conc_pm2_5', 'num_conc_pm4', 'num_conc_pm10', 'particle_size'],
        'join_col': 'measurement_id',
        'description': 'SPS30 particulate matter sensor (PM concentrations and particle counts)'
    }
}

def get_data_date_range(conn):
    """Get the date range from the measurements table"""
    try:
        date_query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM measurements"
        date_df = pd.read_sql_query(date_query, conn)
        
        if not date_df.empty and date_df['min_date'].iloc[0]:
            min_date = pd.to_datetime(date_df['min_date'].iloc[0]).date()
            max_date = pd.to_datetime(date_df['max_date'].iloc[0]).date()
            return min_date, max_date
        else:
            # Fallback dates
            return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date()
    except:
        return datetime(2021, 1, 1).date(), datetime(2022, 12, 31).date()

def show():
    st.header("🔎 SQL Query Interface")
    st.markdown("Execute custom SQL queries on your sensor database.")
    
    db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
    
    if not os.path.exists(db_path):
        st.error("Database file not found. Please check your .env configuration.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table information for reference
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        if not tables:
            st.warning("No tables found in the database.")
            conn.close()
            return
        
        # Get data date range for dynamic options
        min_date, max_date = get_data_date_range(conn)
        
        # Display available tables and schema with sensor-specific information
        with st.expander("📋 Sensor Database Schema", expanded=False):
            st.markdown("**Sensor Database Structure (2021-2022 data):**")
            for table_name, schema in SENSOR_TABLES_SCHEMA.items():
                if table_name in tables:
                    st.markdown(f"**{table_name}:** {schema['description']}")
                    st.markdown(f"Columns: `{', '.join(schema['columns'])}`")
                    if 'datetime_col' in schema:
                        st.markdown(f"🕒 Timestamp column: `{schema['datetime_col']}` (2021-2022 range)")
                    elif 'join_col' in schema:
                        st.markdown(f"🔗 Links to measurements via: `{schema['join_col']}`")
                    st.markdown("---")
        
        # Smart Query Helper for sensor data
        st.subheader("🛠️ Smart Sensor Query Helper")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sensor_tables = [t for t in SENSOR_TABLES_SCHEMA.keys() if t in tables]
            helper_table = st.selectbox("Select sensor table:", ["None"] + sensor_tables, key="helper_table")
        
        with col2:
            if helper_table != "None":
                if helper_table == "measurements":
                    helper_date_col = "timestamp"
                    st.info("Uses timestamp column")
                else:
                    helper_date_col = "timestamp (via join)"
                    st.info("Date filtering via measurements join")
            else:
                helper_date_col = None
                st.info("Select a table first")
        
        with col3:
            if helper_table != "None":
                # Generate dynamic time period options based on actual data
                time_period_options = ["Last 7 Days", "Last Month"]
                
                # Add the most recent complete month if we have data
                if max_date.day > 1:  # Not the first day of month
                    prev_month = max_date.replace(day=1) - timedelta(days=1)
                    time_period_options.append(f"{prev_month.strftime('%b %Y')}")
                
                # Add yearly options for available years
                data_start_year = min_date.year
                data_end_year = max_date.year
                for year in range(data_end_year, data_start_year - 1, -1):
                    time_period_options.append(f"All {year}")
                
                time_period = st.selectbox("Time period:", time_period_options, key="helper_period")
        
        # Generate smart sensor query
        if helper_table != "None" and st.button("Generate Sensor Query"):
            try:
                # Set date range based on period (using max_date as reference)
                ref_date = max_date
                
                if time_period == "Last 7 Days":
                    start_date = ref_date - timedelta(days=7)
                    end_date = ref_date
                elif time_period == "Last Month":
                    start_date = ref_date - timedelta(days=30)
                    end_date = ref_date
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
                        # Fallback to last month if parsing fails
                        start_date = ref_date - timedelta(days=30)
                        end_date = ref_date
                else:
                    # Default fallback
                    start_date = ref_date - timedelta(days=30)
                    end_date = ref_date
                
                if helper_table == "measurements":
                    smart_query = f"""-- Smart Sensor Query: {time_period} from {helper_table}
SELECT * FROM measurements
WHERE timestamp >= '{start_date}'
AND timestamp <= '{end_date} 23:59:59'
ORDER BY timestamp DESC
LIMIT 1000;"""
                else:
                    # Generate join query for sensor tables
                    schema = SENSOR_TABLES_SCHEMA[helper_table]
                    join_col = schema['join_col']
                    
                    smart_query = f"""-- Smart Sensor Query: {time_period} from {helper_table}
SELECT s.*, m.timestamp, m.node_id
FROM {helper_table} s
JOIN measurements m ON s.{join_col} = m.id
WHERE m.timestamp >= '{start_date}'
AND m.timestamp <= '{end_date} 23:59:59'
ORDER BY m.timestamp DESC
LIMIT 1000;"""
                
                st.session_state.smart_query = smart_query
                st.success(f"Generated {helper_table} query for {time_period}")
            except Exception as e:
                st.error(f"Error generating smart query: {str(e)}")
        
        # Pre-defined sensor-specific query examples
        st.subheader("📚 Sensor Query Examples")
        
        example_queries = {
            "Show all tables": "SELECT name FROM sqlite_master WHERE type='table';",
            "Measurements overview": """
-- Recent measurements with node information
SELECT 
    COUNT(*) as total_measurements,
    COUNT(DISTINCT node_id) as unique_nodes,
    MIN(timestamp) as earliest_data,
    MAX(timestamp) as latest_data
FROM measurements;""",
            "BME680 temperature trends": """
-- BME680 temperature data with timestamps
SELECT 
    m.timestamp,
    m.node_id,
    b.temperature,
    b.humidity,
    b.pressure
FROM sensor_bme680 b
JOIN measurements m ON b.measurement_id = m.id
WHERE m.timestamp >= '2022-12-01'
ORDER BY m.timestamp DESC
LIMIT 100;""",
            "SCD30 CO2 analysis": """
-- SCD30 CO2 levels by hour
SELECT 
    strftime('%Y-%m-%d %H:00:00', m.timestamp) as hour,
    AVG(s.co2) as avg_co2,
    MIN(s.co2) as min_co2,
    MAX(s.co2) as max_co2,
    COUNT(*) as readings
FROM sensor_scd30 s
JOIN measurements m ON s.measurement_id = m.id
WHERE m.timestamp >= '2022-12-01'
GROUP BY strftime('%Y-%m-%d %H:00:00', m.timestamp)
ORDER BY hour DESC
LIMIT 24;""",
            "SPS30 air quality": """
-- SPS30 particulate matter concentrations
SELECT 
    m.timestamp,
    m.node_id,
    s.mass_conc_pm2_5,
    s.mass_conc_pm10,
    s.particle_size
FROM sensor_sps30 s
JOIN measurements m ON s.measurement_id = m.id
WHERE m.timestamp >= '2022-12-01'
AND s.mass_conc_pm2_5 IS NOT NULL
ORDER BY m.timestamp DESC
LIMIT 100;""",
            "Multi-sensor comparison": """
-- Compare all sensors for a specific time period
SELECT 
    m.timestamp,
    m.node_id,
    b.temperature as bme680_temp,
    b.humidity as bme680_humidity,
    c.co2,
    c.temperature as scd30_temp,
    p.mass_conc_pm2_5
FROM measurements m
LEFT JOIN sensor_bme680 b ON b.measurement_id = m.id
LEFT JOIN sensor_scd30 c ON c.measurement_id = m.id  
LEFT JOIN sensor_sps30 p ON p.measurement_id = m.id
WHERE m.timestamp >= '2022-12-01'
ORDER BY m.timestamp DESC
LIMIT 50;"""
        }
        
        if tables:
            # Add table-specific examples
            first_table = tables[0]
            first_table_datetime_cols = SENSOR_TABLES_SCHEMA.get(first_table, {}).get('columns', [])
            
            example_queries[f"Sample data from {first_table}"] = f"SELECT * FROM {first_table} LIMIT 10;"
            example_queries[f"Record count for {first_table}"] = f"SELECT COUNT(*) as total_records FROM {first_table};"
            
            # Add date-aware examples if datetime columns exist
            if first_table_datetime_cols:
                first_date_col = first_table_datetime_cols[0]
                example_queries[f"Recent data from {first_table}"] = f"""
-- Recent data from {first_table} (last 24 hours)
SELECT * FROM {first_table}
WHERE {first_date_col} >= datetime('now', '-1 day')
ORDER BY {first_date_col} DESC
LIMIT 100;"""
                
                example_queries[f"Data by hour from {first_table}"] = f"""
-- Hourly aggregation from {first_table}
SELECT 
    strftime('%Y-%m-%d %H:00:00', {first_date_col}) as hour,
    COUNT(*) as record_count
FROM {first_table}
WHERE {first_date_col} >= datetime('now', '-7 days')
GROUP BY strftime('%Y-%m-%d %H:00:00', {first_date_col})
ORDER BY hour DESC;"""
                
                example_queries[f"Date range from {first_table}"] = f"""
-- Custom date range from {first_table}
SELECT * FROM {first_table}
WHERE {first_date_col} >= '2024-01-01'
AND {first_date_col} < '2024-01-02'
ORDER BY {first_date_col}
LIMIT 1000;"""
        
        selected_example = st.selectbox("Choose an example query:", ["Custom Query"] + list(example_queries.keys()))
        
        # Query input
        st.subheader("📝 SQL Query")
        
        # Use smart query if generated
        if 'smart_query' in st.session_state:
            default_query = st.session_state.smart_query
            del st.session_state.smart_query
        elif selected_example != "Custom Query":
            default_query = example_queries[selected_example]
        else:
            default_query = "SELECT * FROM table_name LIMIT 10;"
        
        query = st.text_area(
            "Enter your SQL query:",
            value=default_query,
            height=200,
            help="""Write your SQL query here. Tips:
• Use SELECT statements to query data
• Add WHERE clauses with date ranges for performance
• Use LIMIT to prevent loading too much data
• ORDER BY date columns for time series analysis"""
        )
        
        # Query performance warnings
        if query.strip():
            query_upper = query.upper().strip()
            if 'LIMIT' not in query_upper and 'COUNT' not in query_upper:
                st.warning("⚠️ Consider adding LIMIT to your query to prevent loading too much data")
            
            # Check for date filtering
            has_where = 'WHERE' in query_upper
            has_date_filter = any(col in query for table_cols in SENSOR_TABLES_SCHEMA.values() for col in table_cols)
            
            if not has_where and not has_date_filter:
                st.info("💡 Consider adding date filters (WHERE date_column >= 'YYYY-MM-DD') for better performance")
        
        # Query execution
        col1, col2 = st.columns([1, 4])
        
        with col1:
            execute_query = st.button("Execute Query", type="primary")
        
        with col2:
            if st.button("Clear Results"):
                if 'query_result' in st.session_state:
                    del st.session_state.query_result
        
        # Safety checks and execution
        if execute_query and query.strip():
            query_upper = query.upper().strip()
            
            # Basic safety check - only allow SELECT, PRAGMA, and EXPLAIN statements
            safe_keywords = ['SELECT', 'PRAGMA', 'EXPLAIN', 'WITH']
            is_safe = any(query_upper.startswith(keyword) for keyword in safe_keywords)
            
            if not is_safe:
                st.error("⚠️ Only SELECT, PRAGMA, EXPLAIN, and WITH statements are allowed for safety reasons.")
            else:
                # Performance warnings
                estimated_risk = "LOW"
                if 'LIMIT' not in query_upper and 'COUNT' not in query_upper:
                    estimated_risk = "HIGH"
                    st.warning("⚠️ **HIGH RISK**: Query may return large dataset. Consider adding LIMIT clause.")
                elif 'WHERE' not in query_upper:
                    estimated_risk = "MEDIUM"
                    st.info("💡 **MEDIUM RISK**: Query without filters may be slow. Consider adding WHERE clause.")
                
                try:
                    # Execute query with timeout and size limits
                    with st.spinner(f"Executing query (Risk: {estimated_risk})..."):
                        start_time = pd.Timestamp.now()
                        
                    if query_upper.startswith('SELECT') or query_upper.startswith('WITH'):
                        # Add automatic LIMIT if not present and not aggregation
                        if 'LIMIT' not in query_upper and not any(agg in query_upper for agg in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX', 'GROUP BY']):
                            query_with_limit = query.rstrip(';') + ' LIMIT 10000;'
                            st.info("🛡️ Added automatic LIMIT 10000 for safety")
                            df = pd.read_sql_query(query_with_limit, conn)
                        else:
                            df = pd.read_sql_query(query, conn)
                        
                        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
                        st.session_state.query_result = df
                        st.session_state.execution_time = execution_time
                        
                    else:
                        cursor.execute(query)
                        results = cursor.fetchall()
                        execution_time = (pd.Timestamp.now() - start_time).total_seconds()
                        
                        if results:
                            columns = [str(description[0]) for description in cursor.description] if cursor.description else ['Result']
                            df = pd.DataFrame(results, columns=columns)
                            st.session_state.query_result = df
                        else:
                            st.session_state.query_result = pd.DataFrame({'Result': ['Query executed successfully']})
                        
                        st.session_state.execution_time = execution_time
                
                except Exception as e:
                    st.error(f"❌ **Query Error:** {str(e)}")
                    st.info("💡 **Tips:**\n- Check table and column names\n- Ensure proper SQL syntax\n- Try adding date filters for large tables\n- Use LIMIT for testing queries")
        
        # Display results
        if 'query_result' in st.session_state:
            st.subheader("📊 Query Results")
            
            df = st.session_state.query_result
            execution_time = st.session_state.get('execution_time', 0)
            
            # Performance metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows Returned", f"{len(df):,}")
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
                st.metric("Memory Usage", f"{memory_mb:.1f} MB")
            with col4:
                st.metric("Execution Time", f"{execution_time:.2f}s")
            
            # Action buttons
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("📥 Download CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Click to Download",
                        data=csv,
                        file_name="query_results.csv",
                        mime="text/csv"
                    )
            
            with col_b:
                if st.button("🔄 Clear Results"):
                    if 'query_result' in st.session_state:
                        del st.session_state.query_result
                    if 'execution_time' in st.session_state:
                        del st.session_state.execution_time
                    st.rerun()
            
            with col_c:
                show_preview_only = st.checkbox("Preview Mode (100 rows)", value=len(df) > 100)
            
            # Display the data
            if show_preview_only and len(df) > 100:
                st.info(f"📋 Showing first 100 rows of {len(df):,} total results")
                st.dataframe(df.head(100), use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
            
            # Data analysis section
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Show data types
                with st.expander("📋 Column Information"):
                    col_info = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': [str(df[col].dtype) for col in df.columns],
                        'Non-Null Count': [df[col].count() for col in df.columns],
                        'Unique Values': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
            
            with col_right:
                # Quick statistics for numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    with st.expander("📈 Quick Statistics"):
                        st.write("**Numeric Columns Summary:**")
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        conn.close()
        
    except Exception as e:
        st.error(f"Error with query interface: {str(e)}")
        if 'conn' in locals():
            conn.close() 