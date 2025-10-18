# 📊 SensorScope Analytics

**Comprehensive Multi-Sensor Data Analysis Platform**

SensorScope Analytics is a powerful, user-friendly data analysis platform designed for exploring, analyzing, and visualizing sensor data collected from laboratory and industrial environments. Built with Streamlit, it provides an intuitive interface for data scientists, engineers, and researchers to gain insights from their multi-sensor datasets.

## 🚀 Features

- **🔍 Database Explorer**: Browse and explore multiple database tables with interactive schema viewing
- **📈 Statistical Analysis**: Generate comprehensive statistical insights with automated data profiling
- **🔎 Custom SQL Queries**: Execute custom queries with a safe, user-friendly interface
- **📊 Interactive Visualizations**: Create dynamic charts, graphs, and plots using Plotly
- **🧹 Data Cleaning**: Removing Null values, outlier or irregularities
- **❌ Disruption Analysis**: Looking at Irregular behavour, or Node and Network disruption.
- **👥 Team Collaboration**: Built for 4-person team collaboration with shared database access
- **🛡️ Security**: Safe query execution with built-in SQL injection protection

## 🏗️ Project Structure

```
framework/
├── app.py                 # Main Streamlit application
├── pages/                 # Multi-page application modules
│   ├── __init__.py
│   ├── database_explorer.py
│   ├── table_statistics.py
│   ├── query_interface.py
│   ├── data_visualization.py
│   ├── data_cleaning.py
│   ├── disruption_analysis.py
│   └── team_info.py
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
└── README.md             # This file
```

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- SQLite database file with sensor data

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd framework
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Database Connection

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env file and update the DATABASE_PATH
# Point it to your SQLite database file
nano .env  # or use your preferred editor
```

**Example .env configuration:**
```env
DATABASE_PATH=path/to/your/sensor_data.db
DB_TIMEOUT=30
DB_CHECK_SAME_THREAD=False
```

### Step 5: Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📋 Usage Guide

### Home Dashboard
- View database overview with table counts and statistics
- Quick access to all application features
- Database connection status monitoring

### Database Explorer
- Browse all available tables in your database
- View table schemas and column information
- Sample data preview with customizable row limits
- Interactive table selection and exploration

### Table Statistics
- Comprehensive statistical analysis of selected tables
- Automated data profiling with column information
- Distribution plots and correlation analysis
- Support for both numerical and categorical data

### Query Interface
- Safe SQL query execution with built-in security
- Pre-defined query examples for common operations
- Interactive query builder with table schema reference
- Export query results as CSV files

### Data Visualization
- Interactive chart creation with multiple chart types:
  - Line charts and time series
  - Bar charts and histograms
  - Scatter plots and 3D visualizations
  - Box plots and correlation heatmaps
- Dynamic column selection and filtering
- Responsive, publication-ready charts

### Data Cleaning
- Node_Id based data removal
- Dynamic data cleaning with 3 different approaches:
  - Hardcoded Bounds Data Cleaning: using sensor, domain knowledge
  - SVM-Based Data Cleaning: using hardcoded bounds to train a machine learning model
  - Weighted IQR Data Cleaning: using an interactive filtering mechanism without domain knowledge (useful for most applications).

### Disruption Analysis
- Dynamic user defined disruption analysis.
- Expected Uptime Vs. Downtime analysis.
- Logistic regression based irregularity analysis / node reading malfuntion.

## 🗄️ Database Requirements

Your SQLite database should contain sensor data with the following characteristics:

- **Tables**: Multiple tables representing different sensor types or measurement categories
- **Columns**: Mix of numerical (sensor readings) and categorical (sensor IDs, locations) data
- **Time Series**: Timestamp columns for temporal analysis (recommended)
- **Data Quality**: Clean, consistent data format for optimal analysis

## 👥 Team Collaboration

This platform is designed for collaborative analysis:

1. **Shared Database**: All team members access the same SQLite database
2. **Version Control**: Application code is version controlled for team development
3. **Reproducible Analysis**: Query history and visualization configurations can be shared
4. **Role-Based Access**: Different team members can focus on different analysis aspects

## 🔧 Customization

### Adding New Analysis Pages

1. Create a new Python file in the `pages/` directory
2. Implement a `show()` function with your analysis logic
3. Import and add the page to `app.py`

### Database Configuration

- Modify database connection settings in `.env`
- Adjust timeout and connection parameters as needed
- Support for multiple database files through configuration

### Styling and Branding

- Custom CSS can be added to `app.py`
- Logo and branding elements can be customized
- Theme colors and layouts are configurable

## 🚨 Troubleshooting

### Common Issues

**Database Connection Error**:
- Verify the database file exists at the specified path
- Check file permissions and accessibility
- Ensure the database is not locked by another process

**Import Errors**:
- Confirm all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Verify virtual environment is activated

**Performance Issues**:
- Large datasets may require pagination or sampling
- Consider database indexing for complex queries
- Monitor memory usage for statistical computations

### Getting Help

1. Check the console output for detailed error messages
2. Verify database schema compatibility
3. Review the Team Info page for project context
4. Contact your team members for collaborative debugging

## 📈 Development Roadmap

- [ ] **Enhanced Security**: Advanced user authentication and role management
- [ ] **Real-time Monitoring**: Live sensor data streaming capabilities
- [ ] **Machine Learning**: Integrated ML models for predictive analysis
- [ ] **Report Generation**: Automated report creation and scheduling
- [ ] **Data Export**: Enhanced export capabilities for various formats
- [ ] **Mobile Responsiveness**: Optimized mobile and tablet experience

## 🤝 Contributing

This is a collaborative project developed by a 4-person team:
1. Raj Maharjan (@rajmacurry)
2. Ahmad Ashraf Zargar (ahmadzargar)
3. Zhesan Malik (zshancs)
4. Shreya Jindal

For contributions:

1. Follow the established code structure and naming conventions
2. Test all database interactions thoroughly
3. Document new features and analysis methods
4. Coordinate with team members before major changes

## 📄 License

This project is developed for academic and research purposes as part of a collaborative sensor analysis study.

## 🏆 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the web interface
- Powered by [Plotly](https://plotly.com/) for interactive visualizations
- Database analysis using [Pandas](https://pandas.pydata.org/) and [SQLite](https://sqlite.org/)
- Team collaboration facilitated through modern data science practices

---

**SensorScope Analytics** - *Empowering sensor data analysis through collaborative intelligence* 🚀
