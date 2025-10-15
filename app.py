"""
SensorScope Analytics - Multi-Sensor Data Analysis Platform
Main application entry point for the Streamlit multi-page app
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sqlite3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SensorScope Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
    }
    .metric-card {
        background-color: #d0d2d6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    /* Fix sidebar visibility */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* Ensure sidebar text is visible */
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
        color: #262730;
    }
    
    /* Fix option menu text visibility */
    .nav-link {
        color: #262730 !important;
    }
    
    .nav-link:hover {
        color: #1f77b4 !important;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<h1 class="main-header">📊 SensorScope Analytics</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comprehensive Multi-Sensor Data Analysis Platform</p>', unsafe_allow_html=True)
    
    # Navigation menu - ALWAYS render this first
    with st.sidebar:
        # App branding
        st.markdown("# 📊 SensorScope")
        st.markdown("*Analytics Platform*")
        st.markdown("---")
        
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Database Explorer", "Table Statistics", "Query Interface", "Data Visualization", "Data Cleaning", "Disruption Analysis", "Task Analysis", "co2 Prediction","Team Info"],
            icons=["house", "database", "bar-chart", "terminal", "graph-up", "gear", "exclamation","search", "cpu", "people"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important", 
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "#1f77b4", 
                    "font-size": "20px"
                },
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "5px",
                    "padding": "10px 15px",
                    "--hover-color": "#e6f3ff",
                    "color": "#262730",
                    "background-color": "#d0d2ee",
                    # "border-radius": "5px"
                },
                "nav-link-selected": {
                    "background-color": "#1f77b4",
                    "color": "white",
                    "font-weight": "bold"
                },
            }
        )
        
        # Debug information in collapsible section
        with st.expander("🔧 Debug Info", expanded=False):
            db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
            st.write(f"**DB Path:** `{db_path}`")
            st.write(f"**DB Exists:** {os.path.exists(db_path)}")
            st.write(f"**Working Dir:** `{os.getcwd()}`")
    
    # Database connection check (but don't prevent navigation)
    if not os.path.exists(db_path) and selected == "Home":
        st.warning(f"""
        ⚠️ **Database not found at:** `{db_path}`
        
        Please ensure your SQLite database file exists and the path is correctly set in your `.env` file.
        """)
        st.info("""
        **To get started:**
        1. Place your SQLite database file in the project directory
        2. Update the `DATABASE_PATH` in your `.env` file
        3. Refresh this page
        """)
        
        # Show current working directory for debugging
        st.code(f"Current working directory: {os.getcwd()}")
        st.code(f"Files in current directory: {', '.join(os.listdir('.'))}")
    
    # Route to different pages based on selection
    if selected == "Home":
        show_home()
    elif selected == "Database Explorer":
        show_database_explorer()
    elif selected == "Table Statistics":
        show_table_statistics()
    elif selected == "Query Interface":
        show_query_interface()
    elif selected == "Data Visualization":
        show_data_visualization()
    elif selected == "Data Cleaning":
        show_data_cleaning()
    elif selected == "Disruption Analysis":
        show_disruption_analysis()
    elif selected == "Task Analysis":
        show_task_analysis()
    elif selected == "co2 Prediction":
        show_co2_prediction()
    elif selected == "Team Info":
        show_team_info()

def show_home():
    st.markdown("## Welcome to SensorScope Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Explore Data</h3>
            <p>Browse through multiple sensor data tables and understand your data structure</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Analyze Statistics</h3>
            <p>Get comprehensive statistical insights about your sensor data tables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔎 Custom Queries</h3>
            <p>Run custom SQL queries to extract specific insights from your data</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick database info
    try:
        db_path = os.getenv('DATABASE_PATH', 'sensor_data.db')
        print(f"Database path: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table count
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        table_count = len(tables)
        
        # Get total record count across all tables
        total_records = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            total_records += count
        
        conn.close()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Database Tables", table_count)
        with col2:
            st.metric("Total Records", f"{total_records:,}")
        with col3:
            st.metric("Database Size", f"{os.path.getsize(db_path) / (1024*1024):.1f} MB")
        with col4:
            st.metric("Team Members", "4")
            
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")

def show_database_explorer():
    try:
        from modules import database_explorer
        database_explorer.show()
    except Exception as e:
        st.error(f"Error loading Database Explorer: {str(e)}")
        st.code(str(e))

def show_table_statistics():
    try:
        from modules import table_statistics  
        table_statistics.show()
    except Exception as e:
        st.error(f"Error loading Table Statistics: {str(e)}")
        st.code(str(e))

def show_query_interface():
    try:
        from modules import query_interface
        query_interface.show()
    except Exception as e:
        st.error(f"Error loading Query Interface: {str(e)}")
        st.code(str(e))

def show_data_visualization():
    try:
        from modules import data_visualization
        data_visualization.show()
    except Exception as e:
        st.error(f"Error loading Data Visualization: {str(e)}")
        st.code(str(e))

def show_data_cleaning():
    try:
        from modules import data_cleaning
        data_cleaning.show()
    except Exception as e:
        st.error(f"Error loading Data Cleaning: {str(e)}")
        st.code(str(e))

def show_disruption_analysis():
    try:
        from modules import disruption_analysis
        disruption_analysis.show()
    except Exception as e:
        st.error(f"Error loading Disruption Analysis: {str(e)}")
        st.code(str(e))

def show_task_analysis():
    try:
        from modules import task_analysis
        task_analysis.show()
    except Exception as e:
        st.error(f"Error loading Task Analysis: {str(e)}")
        st.code(str(e))

def show_co2_prediction():
    try:
        from modules import co2_prediction
        co2_prediction.show()
    except Exception as e:
        st.error(f"Error loading co2 Prediction: {str(e)}")
        st.code(str(e))

def show_team_info():
    try:
        from modules import team_info
        team_info.show()
    except Exception as e:
        st.error(f"Error loading Team Info: {str(e)}")
        st.code(str(e))

if __name__ == "__main__":
    main() 