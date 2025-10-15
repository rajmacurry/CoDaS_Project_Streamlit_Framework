"""
Team Information Page - Information about the project team
"""

import streamlit as st

def show():
    st.header("👥 Team Information")
    st.markdown("Meet the team behind SensorScope Analytics!")
    
    # Debug - ensure the page loads
    st.write("✅ Team Info page loaded successfully")
    
    # Project overview
    st.subheader("🎯 Project Overview")
    st.markdown("""
    **SensorScope Analytics** is a comprehensive data analysis platform developed as part of a collaborative 
    4-person team effort to study sensor data collected from multiple sensors in laboratory and industrial environments.
    
    Our platform provides powerful tools for:
    - **Database Exploration**: Browse and understand your sensor data structure
    - **Statistical Analysis**: Generate comprehensive insights about your data
    - **Custom Queries**: Execute SQL queries for specific data extraction
    - **Interactive Visualizations**: Create dynamic charts and graphs
    """)
    
    # Team section
    st.subheader("🚀 Our Team")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Ahmed Ashraf
        - **Focus**: Database architecture and optimization
        - **Skills**: SQL, Database Design, Data Modeling
        """)
        
        st.markdown("""
        #### Malik Zeeshan Ahmad
        - **Focus**: Statistical analysis and machine learning, User Interface and Experience Design
        - **Skills**: Python, Statistics, Data Analysis
        """)
    
    with col2:
        st.markdown("""
        #### Raj Maharjan
        - **Focus**: Data Engineering, Data Pipeline, Interactive Dashboards
        - **Skills**: Streamlit, UI/UX, Data Visualization
        """)
        
        st.markdown("""
        #### Shreya
        - **Focus**: Data Engineering, Data Pipeline
        - **Skills**: Data Engineering, Plotly, Dashboard Design
        """)
    
    st.markdown("---")
    
    # Technology stack
    st.subheader("🛠️ Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend & UI:**
        - Streamlit
        - HTML/CSS
        - Streamlit Components
        """)
    
    with col2:
        st.markdown("""
        **Data Processing:**
        - Pandas
        - NumPy
        - SQLite
        - SQL Alchemy
        """)
    
    with col3:
        st.markdown("""
        **Visualization:**
        - Plotly
        - Matplotlib
        - Seaborn
        """)
    
    # Project goals
    st.subheader("🎯 Project Goals")
    
    goals = [
        "Develop a comprehensive sensor data analysis platform",
        "Enable easy exploration of multi-sensor datasets",
        "Provide statistical insights for decision making",
        "Create interactive visualizations for data understanding",
        "Build a scalable solution for industrial sensor monitoring"
    ]
    
    for i, goal in enumerate(goals, 1):
        st.markdown(f"{i}. {goal}")
    
    st.markdown("---")
    
    # Contact and collaboration
    st.subheader("📞 Collaboration & Contact")
    
    st.markdown("""
    This project is part of an academic collaboration focused on advancing sensor data analysis 
    in laboratory and industrial environments. 
    
    **Key Features of Our Collaboration:**
    - Multi-disciplinary approach combining data science, engineering, and design
    - Agile development methodology
    - Focus on real-world industrial applications
    - Emphasis on user-friendly interfaces for technical and non-technical users
    """)
    
    # Project timeline
    st.subheader("📅 Project Timeline")
    
    timeline_data = {
        "Phase": ["Planning", "Database Setup", "Core Development", "Visualization", "Testing & Deployment"],
        "Status": ["✅ Complete", "✅ Complete", "🟡 In Progress", "🟡 In Progress", "🔄 Upcoming"],
        "Description": [
            "Project planning and team organization",
            "Database schema design and initial setup",
            "Core functionality development",
            "Interactive visualization features",
            "Testing, optimization, and deployment"
        ]
    }
    
    import pandas as pd
    timeline_df = pd.DataFrame(timeline_data)
    st.dataframe(timeline_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("*Thank you for using SensorScope Analytics! 🚀*") 