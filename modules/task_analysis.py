"""
Task Analysis Page - Sensor Data Clustering and Pattern Analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.stats import zscore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define tab10 colors to match matplotlib's tab10 colormap
TAB10_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def create_cluster_color_map(unique_clusters):
    """Create a consistent color mapping for clusters."""
    color_map = {}
    for i, cluster_id in enumerate(sorted(unique_clusters)):
        color_map[str(cluster_id)] = TAB10_COLORS[i % len(TAB10_COLORS)]
    return color_map


def detect_process_days(df):
    """
    Detect which days have useful processes going on vs baseline days.

    Args:
        df: DataFrame with timestamp index and sensor columns

    Returns:
        enhanced_daily: Enhanced daily summary with cluster labels
        daily_mean: Daily means with cluster labels
        process_days: Dates identified as process days
        baseline_days: Dates identified as baseline days
    """
    df = df.copy()
    df = df.dropna()

    # Reset index to make timestamp a column if it's currently the index
    if df.index.name == "timestamp" or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

    # Ensure timestamp is datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        raise ValueError("DataFrame must have a 'timestamp' column")

    # Extract time features
    df["hour"] = df["timestamp"].dt.hour
    df["working_hours"] = df["hour"].between(8, 18)

    # Map database column names to expected feature names
    column_mapping = {
        "bme_temp": "temperature",
        "bme_hum": "humidity",
        "mass_conc_pm2_5": "mass_conc_pm2_5",
        "mass_conc_pm10": "mass_conc_pm10",
        "num_conc_pm2_5": "num_conc_pm2_5",
        "num_conc_pm10": "num_conc_pm10",
        "particle_size": "particle_size",
        "gas_resistance": "gas_resistance",
        "co2": "co2",
    }

    # Rename columns to match expected names
    df = df.rename(columns=column_mapping)

    # Columns to normalize - use available columns
    potential_features = [
        "gas_resistance",
        "co2",
        "temperature",
        "humidity",
        "mass_conc_pm2_5",
        "mass_conc_pm10",
        "num_conc_pm2_5",
        "num_conc_pm10",
        "particle_size",
    ]

    # Only use features that exist in the dataframe
    features_to_scale = [f for f in potential_features if f in df.columns]

    if len(features_to_scale) == 0:
        raise ValueError(
            f"No usable features found. Available columns: {list(df.columns)}"
        )

    # Time columns to keep
    time_columns = ["timestamp", "hour", "working_hours"]

    # Scale features
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[features_to_scale])
    df_scaled_features = pd.DataFrame(
        scaled_values, columns=features_to_scale, index=df.index
    )

    # Filter out outliers (z-score < 3)
    z_scores = df_scaled_features.apply(zscore)
    keep_rows = (z_scores.abs() < 3).all(axis=1)

    df_scaled = pd.concat([df[time_columns], df_scaled_features], axis=1)
    df_no_outliers = df_scaled[keep_rows]
    df_no_outliers["date"] = df_no_outliers["timestamp"].dt.date

    # Enhanced daily summary
    def summarize_day(group):
        working = group[group["working_hours"]]
        non_working = group[~group["working_hours"]]

        summary = {}
        for feature in features_to_scale:
            summary[f"{feature}_working_mean"] = working[feature].mean()
            summary[f"{feature}_non_working_mean"] = non_working[feature].mean()
            summary[f"{feature}_working_std"] = working[feature].std()
            summary[f"{feature}_non_working_std"] = non_working[feature].std()
        return pd.Series(summary)

    enhanced_daily = df_no_outliers.groupby("date").apply(summarize_day)
    daily_mean = df_no_outliers.groupby("date")[features_to_scale].mean()

    # Clustering - use 2 clusters to separate process vs baseline days
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    numeric_cols = enhanced_daily.select_dtypes(include=["number"]).columns

    # Fill NaN values with column means
    enhanced_daily[numeric_cols] = enhanced_daily[numeric_cols].fillna(
        enhanced_daily[numeric_cols].mean()
    )

    # Fit clustering
    clusters = kmeans.fit_predict(enhanced_daily[numeric_cols])

    enhanced_daily["cluster"] = clusters
    daily_mean["cluster"] = clusters

    # Label clusters based on CO2 levels during working hours
    cluster_summary = enhanced_daily.groupby("cluster").mean()

    # Use CO2 if available, otherwise use the first available feature
    if "co2_working_mean" in cluster_summary.columns:
        primary_feature = "co2_working_mean"
    else:
        primary_feature = f"{features_to_scale[0]}_working_mean"

    # Higher values during working hours typically indicate process activity
    if (
        cluster_summary[primary_feature].iloc[0]
        < cluster_summary[primary_feature].iloc[1]
    ):
        cluster_mapping = {0: "baseline", 1: "process"}
    else:
        cluster_mapping = {0: "process", 1: "baseline"}

    enhanced_daily["label"] = enhanced_daily["cluster"].map(cluster_mapping)
    daily_mean["label"] = daily_mean["cluster"].map(cluster_mapping)

    # Extract dates
    process_days = enhanced_daily[enhanced_daily["label"] == "process"].index
    baseline_days = enhanced_daily[enhanced_daily["label"] == "baseline"].index

    return enhanced_daily, daily_mean, process_days, baseline_days


def show():
    st.header("🔍 Task Analysis")
    st.markdown(
        "Analyze and cluster sensor data for any specified time period with advanced filtering options."
    )

    # Main page controls
    st.subheader("⚙️ Analysis Parameters.")

    # Create columns for better layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📅 Time Period**")
        start_date = st.date_input(
            "Start Date",
            value=datetime(2022, 7, 1),
            help="Select the starting date for analysis",
        )

        num_days = st.number_input(
            "Number of Days",
            min_value=1,
            max_value=365,
            value=7,
            help="Number of days to analyze from the start date",
        )

    with col2:
        st.markdown("**🔬 Clustering Settings**")
        clusters = st.slider(
            "Number of Clusters",
            min_value=2,
            max_value=12,
            value=8,
            help="Number of clusters to create",
        )

    with col3:
        st.markdown("**🔧 Filtering Options**")
        working_hours_only = st.checkbox(
            "Working Hours Only (9 AM - 5 PM)",
            value=False,
            help="Only analyze data during working hours",
        )

        exclude_weekends = st.checkbox(
            "Exclude Weekends", value=False, help="Exclude Saturday and Sunday data"
        )

    # Analysis button (centered)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
            st.session_state.analysis_params = {
                "start_date": start_date.strftime("%Y-%m-%d"),
                "num_days": num_days,
                "clusters": clusters,
                "working_hours_only": working_hours_only,
                "exclude_weekends": exclude_weekends,
                "detect_process_days_enabled": True,
            }

    st.divider()

    # Display current parameters
    st.subheader("📊 Current Analysis Parameters")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Start Date", start_date.strftime("%Y-%m-%d"))
        st.metric("Number of Days", num_days)

    with col2:
        st.metric("Number of Clusters", clusters)
        end_date = start_date + timedelta(days=num_days - 1)
        st.metric("End Date", end_date.strftime("%Y-%m-%d"))

    with col3:
        st.metric("Working Hours Only", "Yes" if working_hours_only else "No")
        st.metric("Exclude Weekends", "Yes" if exclude_weekends else "No")
        st.metric("Detect Process Days", "Yes")

    # Run analysis if button was clicked
    if st.session_state.get("run_analysis", False):
        try:
            params = st.session_state.analysis_params
            results = get_clusters_for_period(
                start_date=params["start_date"],
                num_days=params["num_days"],
                clusters=params["clusters"],
                working_hours_only=params["working_hours_only"],
                exclude_weekends=params["exclude_weekends"],
                detect_process_days_enabled=params["detect_process_days_enabled"],
            )

            if results[0] is not None:
                df_pca, df_scaled, df_for_clustering, kmeans, pca, process_results, sensor_scaler, feature_names = results

                # Store results in session state
                st.session_state.analysis_results = {
                    "df_pca": df_pca,
                    "df_scaled": df_scaled,
                    "df_for_clustering": df_for_clustering,
                    "kmeans": kmeans,
                    "pca": pca,
                    "process_results": process_results,
                    "params": params,
                    "scaler": sensor_scaler,
                    "feature_names": feature_names,
                }

                st.success("✅ Analysis completed successfully!")

                # Display results
                display_analysis_results(
                    df_pca, df_scaled, df_for_clustering, kmeans, pca, params, process_results
                )
                
                # Add real-time cluster prediction interface
                display_realtime_prediction_interface(
                    kmeans, sensor_scaler, feature_names, params
                )
            else:
                st.error(
                    "❌ No data found for the specified parameters. Please adjust your selection."
                )

        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")
            st.exception(e)

        # Reset the analysis trigger
        st.session_state.run_analysis = False

    # Display previous results if available
    elif "analysis_results" in st.session_state:
        st.info(
            "📋 Showing previous analysis results. Use the sidebar to run a new analysis."
        )
        results = st.session_state.analysis_results
        display_analysis_results(
            results["df_pca"],
            results["df_scaled"],
            results.get("df_for_clustering", results.get("df_temporal")),  # backward compatibility
            results["kmeans"],
            results["pca"],
            results["params"],
            results.get("process_results", None),
        )
        
        # Add real-time cluster prediction interface
        display_realtime_prediction_interface(
            results["kmeans"],
            results.get("scaler"),
            results.get("feature_names"),
            results["params"]
        )
    else:
        st.info(
            "👆 Use the sidebar controls to configure your analysis parameters and click 'Run Analysis' to begin."
        )


def get_clusters_for_period(
    start_date,
    num_days=7,
    clusters=10,
    working_hours_only=False,
    exclude_weekends=False,
    detect_process_days_enabled=False,
):
    """
    Analyze and cluster sensor data for any specified number of days starting from the given date.
    """

    # Calculate end date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = start_dt + timedelta(days=num_days - 1)
    end_date = end_dt.strftime("%Y-%m-%d")

    st.subheader("🔄 Analysis Process")
    st.write(f"📅 Analyzing data for {num_days} days: {start_date} to {end_date}")
    st.write(
        f"🗓️ Start day: {start_dt.strftime('%A')}, End day: {end_dt.strftime('%A')}"
    )

    # Step 1: Data Loading and Query Construction
    with st.expander("📊 Step 1: Data Loading and Query Construction", expanded=False):
        st.write("**Building database query with specified conditions:**")

        # Build query conditions
        conditions = []

        # Date range condition
        conditions.append(
            f"strftime('%Y-%m-%d', m.timestamp) BETWEEN '{start_date}' AND '{end_date}'"
        )
        st.write(f"• Date range: {start_date} to {end_date}")

        # Working hours condition
        if working_hours_only:
            st.write("• Working hours filter: 9 AM - 5 PM")
            conditions.append("strftime('%H', m.timestamp) BETWEEN '09' AND '17'")
        else:
            st.write("• Working hours filter: Not applied")

        # Weekend exclusion condition
        if exclude_weekends:
            st.write("• Weekend exclusion: Saturday and Sunday excluded")
            conditions.append("strftime('%w', m.timestamp) NOT IN ('0', '6')")
        else:
            st.write("• Weekend exclusion: Not applied")

        # Combine all conditions
        where_clause = " AND ".join(conditions)

        # Features to extract
        features_to_use = [
            "bme.gas_resistance",
            "bme.pressure",
            "bme.temperature AS bme_temp",
            "bme.humidity AS bme_hum",
            "scd.co2",
            "sps.mass_conc_pm1",
            "sps.mass_conc_pm2_5",
            "sps.mass_conc_pm4",
            "sps.mass_conc_pm10",
            "sps.num_conc_pm0_5",
            "sps.num_conc_pm1",
            "sps.num_conc_pm2_5",
            "sps.num_conc_pm4",
            "sps.num_conc_pm10",
            "sps.particle_size",
        ]
        features_joined = ", ".join(features_to_use)

        st.write(f"• Features to extract: {len(features_to_use)} sensor features")
        st.write("• Query conditions:", where_clause)

        # Query for the specified period
        query = f"""
        SELECT m.timestamp,
            {features_joined}
        FROM measurements m
        LEFT JOIN sensor_bme680 bme ON m.id = bme.measurement_id
        LEFT JOIN sensor_scd30 scd ON m.id = scd.measurement_id
        LEFT JOIN sensor_sps30 sps ON m.id = sps.measurement_id
        WHERE {where_clause};
        """

        # Connect to database
        db_path = os.getenv("DATABASE_PATH", "sensor_data.db")
        conn = sqlite3.connect(db_path)

        try:
            with st.spinner("Loading data from database..."):
                df = pd.read_sql(query, conn)

            if df.empty:
                st.error("❌ No data found for the specified period and conditions!")
                return None, None, None, None, None

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

            st.success(
                f"✅ Data loaded successfully: {df.shape[0]:,} records with {df.shape[1]} features"
            )
            st.write(f"📈 Data shape: {df.shape}")
            st.write(f"📊 Available features: {list(df.columns)}")

        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None, None, None, None, None
        finally:
            conn.close()

    # Step 2: Data Preprocessing
    with st.expander("🧹 Step 2: Data Preprocessing", expanded=False):
        st.write("**Cleaning and preparing data for analysis:**")

        # Remove missing values
        df_analysis = df.copy()
        original_shape = df_analysis.shape
        df_analysis = df_analysis.dropna()

        removed_rows = original_shape[0] - df_analysis.shape[0]
        st.write(f"• Original data points: {original_shape[0]:,}")
        st.write(f"• Removed missing values: {removed_rows:,} rows")
        st.write(f"• Remaining data points: {df_analysis.shape[0]:,}")

        if df_analysis.empty:
            st.error("❌ No valid data remaining after removing missing values!")
            return None, None, None, None, None

        # Feature scaling
        features = df_analysis.columns
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df_analysis), columns=features, index=df_analysis.index
        )
        
        # Store scaler for later use
        sensor_scaler = scaler

        st.success(f"✅ Data preprocessing completed")
        st.write(f"• Features standardized using StandardScaler")
        st.write(f"• All features now have mean=0 and std=1")

    # Step 3: Outlier Detection and Removal
    with st.expander("🎯 Step 3: Outlier Detection and Removal", expanded=False):
        st.write("**Detecting and removing outliers using z-score method:**")

        # Remove outliers using z-score
        z_scores = df_scaled.apply(zscore)
        outlier_mask = (z_scores.abs() >= 3).any(axis=1)
        df_no_outliers = df_scaled[~outlier_mask]

        outliers_removed = df_scaled.shape[0] - df_no_outliers.shape[0]
        outlier_percentage = (outliers_removed / df_scaled.shape[0]) * 100

        st.write(f"• Outlier detection threshold: |z-score| >= 3")
        st.write(f"• Data points before outlier removal: {df_scaled.shape[0]:,}")
        st.write(
            f"• Outliers detected and removed: {outliers_removed:,} ({outlier_percentage:.1f}%)"
        )
        st.write(f"• Data points after outlier removal: {df_no_outliers.shape[0]:,}")

        df_scaled = df_no_outliers

        if df_scaled.empty:
            st.error("❌ No data remaining after outlier removal!")
            return None, None, None, None, None

        st.success(f"✅ Outlier removal completed")

    # Step 4: Prepare Data for Clustering
    with st.expander("🔧 Step 4: Prepare Data for Clustering", expanded=False):
        st.write("**Preparing sensor data for clustering analysis:**")

        # Use only sensor features for clustering
        df_for_clustering = df_scaled.copy()

        st.write(f"• Using pure sensor data for clustering")
        st.write(f"• No temporal features added - letting data speak for itself")
        st.write(f"• Total features for clustering: {df_for_clustering.shape[1]}")
        st.write(f"• Features: {list(df_for_clustering.columns)}")

        st.success(f"✅ Data preparation completed")

    # Step 5: Principal Component Analysis (PCA)
    with st.expander("🔬 Step 5: Principal Component Analysis (PCA)", expanded=False):
        st.write("**Reducing dimensionality using PCA for visualization:**")

        # PCA for dimensionality reduction
        n_components = min(3, len(df_scaled.columns))
        pca = PCA(n_components=n_components)
        sensor_features = [col for col in df_scaled.columns if col in features]
        df_pca = pd.DataFrame(
            pca.fit_transform(df_scaled[sensor_features]),
            columns=[f"PC{i+1}" for i in range(n_components)],
            index=df_scaled.index,
        )

        explained_variance = pca.explained_variance_ratio_
        total_variance = sum(explained_variance)

        st.write(f"• Number of principal components: {n_components}")
        st.write(f"• Sensor features used for PCA: {len(sensor_features)}")
        for i, var in enumerate(explained_variance):
            st.write(f"  - PC{i+1}: {var:.3f} ({var*100:.1f}% of variance)")
        st.write(
            f"• Total variance explained: {total_variance:.3f} ({total_variance*100:.1f}%)"
        )

        st.success(f"✅ PCA completed")

    # Step 6: K-Means Clustering
    with st.expander("🎯 Step 6: K-Means Clustering", expanded=False):
        st.write("**Performing K-Means clustering on pure sensor data:**")

        # Clustering using only sensor data
        clustering_features = df_for_clustering
        kmeans = KMeans(n_clusters=clusters, random_state=42, n_init=10)

        with st.spinner("Running K-Means clustering..."):
            cluster_labels = kmeans.fit_predict(clustering_features)

        df_pca["cluster"] = cluster_labels.astype(str)

        # Cluster statistics
        cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()

        st.write(f"• Number of clusters: {clusters}")
        st.write(f"• Features used for clustering: {clustering_features.shape[1]} (sensor only)")
        st.write(f"• Clustering algorithm: K-Means with random_state=42")
        st.write(f"• Cluster distribution:")
        for cluster_id, count in cluster_counts.items():
            percentage = (count / len(cluster_labels)) * 100
            st.write(f"  - Cluster {cluster_id}: {count:,} points ({percentage:.1f}%)")

        st.success(f"✅ K-Means clustering completed")

    # Step 7: Process Day Detection (always enabled)
    process_results = None
    if detect_process_days_enabled:
        with st.expander("📅 Step 7: Process Day Detection", expanded=False):
            st.write("**Detecting days with significant process activity:**")

            try:
                # Get the original data for process day detection
                original_query = f"""
                SELECT m.timestamp,
                    {features_joined}
                FROM measurements m
                LEFT JOIN sensor_bme680 bme ON m.id = bme.measurement_id
                LEFT JOIN sensor_scd30 scd ON m.id = scd.measurement_id
                LEFT JOIN sensor_sps30 sps ON m.id = sps.measurement_id
                WHERE {where_clause};
                """

                db_path = os.getenv("DATABASE_PATH", "sensor_data.db")
                conn = sqlite3.connect(db_path)

                try:
                    original_df = pd.read_sql(original_query, conn)
                    original_df["timestamp"] = pd.to_datetime(original_df["timestamp"])

                    # Run process day detection
                    enhanced_daily, daily_mean, process_days, baseline_days = (
                        detect_process_days(original_df)
                    )

                    process_results = {
                        "enhanced_daily": enhanced_daily,
                        "daily_mean": daily_mean,
                        "process_days": process_days,
                        "baseline_days": baseline_days,
                    }

                    st.success("✅ Process day detection completed")

                except Exception as e:
                    st.error(f"❌ Error in process day detection: {str(e)}")

                finally:
                    conn.close()

            except Exception as e:
                st.error(f"❌ Error setting up process day detection: {str(e)}")

    st.success("🎉 All analysis steps completed successfully!")

    return df_pca, df_scaled, df_for_clustering, kmeans, pca, process_results, sensor_scaler, features


def display_analysis_results(
    df_pca, df_scaled, df_temporal, kmeans, pca, params, process_results=None
):
    """Display the analysis results using Streamlit components."""

    st.subheader("📊 Analysis Results")

    # Create consistent color mapping for all charts
    unique_clusters = df_pca["cluster"].unique()
    cluster_color_map = create_cluster_color_map(unique_clusters)

    # Display all analysis sections in sequence
    create_pca_visualizations(df_pca, pca, params, cluster_color_map)

    st.divider()

    create_timeline_visualizations(df_pca, params, cluster_color_map)

    st.divider()

    create_pattern_analysis(df_pca, df_temporal, params, cluster_color_map)

    st.divider()

    # Get feature names for detailed analysis
    feature_names = [
        col
        for col in df_scaled.columns
        if col not in ["hour", "day_of_week", "is_weekend", "day_of_period"]
    ]

    # Detailed cluster characteristics analysis
    cluster_stats = analyze_cluster_characteristics_detailed(
        df_pca, df_scaled, df_temporal, kmeans, feature_names, cluster_color_map
    )

    st.divider()

    create_summary_analysis(df_pca, df_temporal, params)
    
    # Process Day Detection Results (at the end)
    if process_results is not None:
        st.divider()
        
        with st.expander("📅 Process Day Detection Results", expanded=False):
            enhanced_daily = process_results["enhanced_daily"]
            process_days = process_results["process_days"]
            baseline_days = process_results["baseline_days"]
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Days", len(enhanced_daily))
            
            with col2:
                process_percentage = (len(process_days) / len(enhanced_daily)) * 100
                st.metric("Process Days", f"{len(process_days)} ({process_percentage:.1f}%)")
            
            with col3:
                baseline_percentage = (len(baseline_days) / len(enhanced_daily)) * 100
                st.metric("Baseline Days", f"{len(baseline_days)} ({baseline_percentage:.1f}%)")
            
            # Display process days
            if len(process_days) > 0:
                st.subheader("🔥 Process Days")
                process_dates_with_days = []
                for date in sorted(process_days):
                    # Convert date to datetime to get day name
                    date_obj = pd.to_datetime(date)
                    day_name = date_obj.strftime('%A')
                    process_dates_with_days.append(f"{date} ({day_name})")
                st.write(", ".join(process_dates_with_days))

            # Display baseline days
            if len(baseline_days) > 0:
                st.subheader("📊 Baseline Days")
                baseline_dates_with_days = []
                for date in sorted(baseline_days):
                    # Convert date to datetime to get day name
                    date_obj = pd.to_datetime(date)
                    day_name = date_obj.strftime('%A')
                    baseline_dates_with_days.append(f"{date} ({day_name})")
                st.write(", ".join(baseline_dates_with_days))
                
            st.info("💡 **Process days** represent periods with significant sensor activity patterns, while **baseline days** represent normal background conditions.")


def create_pca_visualizations(df_pca, pca, params, cluster_color_map):
    """Create PCA scatter plots using Plotly."""

    st.subheader("🔬 Principal Component Analysis")

    n_components = len([col for col in df_pca.columns if col.startswith("PC")])

    # Ensure consistent cluster ordering
    sorted_clusters = sorted(df_pca["cluster"].unique())

    if n_components >= 2:
        col1, col2 = st.columns(2)

        with col1:
            # PC1 vs PC2
            fig = px.scatter(
                df_pca,
                x="PC1",
                y="PC2",
                color="cluster",
                title="Sensor States (PC1 vs PC2)",
                color_discrete_map=cluster_color_map,
                category_orders={"cluster": sorted_clusters},
                hover_data={"cluster": True},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            if n_components >= 3:
                # PC1 vs PC3
                fig = px.scatter(
                    df_pca,
                    x="PC1",
                    y="PC3",
                    color="cluster",
                    title="Sensor States (PC1 vs PC3)",
                    color_discrete_map=cluster_color_map,
                    category_orders={"cluster": sorted_clusters},
                    hover_data={"cluster": True},
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

        if n_components >= 3:
            # PC2 vs PC3
            fig = px.scatter(
                df_pca,
                x="PC2",
                y="PC3",
                color="cluster",
                title="Sensor States (PC2 vs PC3)",
                color_discrete_map=cluster_color_map,
                category_orders={"cluster": sorted_clusters},
                hover_data={"cluster": True},
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

    # PCA explained variance
    st.subheader("📊 PCA Explained Variance")
    col1, col2 = st.columns(2)

    with col1:
        # Explained variance ratio
        explained_var = pca.explained_variance_ratio_
        fig = px.bar(
            x=[f"PC{i+1}" for i in range(len(explained_var))],
            y=explained_var,
            title="Explained Variance Ratio by Component",
            labels={"x": "Principal Component", "y": "Explained Variance Ratio"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cumulative explained variance
        cumulative_var = np.cumsum(explained_var)
        fig = px.line(
            x=[f"PC{i+1}" for i in range(len(cumulative_var))],
            y=cumulative_var,
            title="Cumulative Explained Variance",
            labels={"x": "Principal Component", "y": "Cumulative Explained Variance"},
        )
        fig.add_hline(
            y=0.90, line_dash="dash", line_color="red", annotation_text="90% Threshold"
        )
        st.plotly_chart(fig, use_container_width=True)


def create_timeline_visualizations(df_pca, params, cluster_color_map):
    """Create timeline visualizations."""

    st.subheader("⏱️ Timeline Analysis")

    # Ensure consistent cluster ordering
    sorted_clusters = sorted(df_pca["cluster"].unique())

    # Cluster timeline
    fig = px.scatter(
        df_pca.reset_index(),
        x="timestamp",
        y="cluster",
        color="cluster",
        title=f"Process States Over {params['num_days']} Days ({params['start_date']} to {(datetime.strptime(params['start_date'], '%Y-%m-%d') + timedelta(days=params['num_days']-1)).strftime('%Y-%m-%d')})",
        color_discrete_map=cluster_color_map,
        category_orders={"cluster": sorted_clusters},
        hover_data={"cluster": True},
    )
    fig.update_layout(height=400, xaxis_title="Time", yaxis_title="Cluster ID")
    st.plotly_chart(fig, use_container_width=True)

    # Daily cluster distribution
    df_daily = df_pca.copy()
    df_daily["date"] = df_daily.index.date
    daily_clusters = (
        df_daily.groupby(["date", "cluster"]).size().reset_index(name="count")
    )

    fig = px.bar(
        daily_clusters,
        x="date",
        y="count",
        color="cluster",
        title="Cluster Distribution by Day",
        color_discrete_map=cluster_color_map,
        category_orders={"cluster": sorted_clusters},
    )
    fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)


def create_pattern_analysis(df_pca, df_temporal, params, cluster_color_map):
    """Create pattern analysis visualizations."""

    st.subheader("📊 Pattern Analysis")

    # Prepare data for analysis
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df_analysis = df_pca.copy()
    timestamps = pd.to_datetime(df_analysis.index)
    df_analysis["day_of_week"] = timestamps.dayofweek
    df_analysis["day_name"] = df_analysis["day_of_week"].map(lambda x: day_names[x])
    df_analysis["hour"] = timestamps.hour
    df_analysis["date"] = timestamps.date

    # Ensure consistent cluster ordering
    sorted_clusters = sorted(df_analysis["cluster"].unique())
    color_sequence = [cluster_color_map[str(cluster)] for cluster in sorted_clusters]

    col1, col2 = st.columns(2)

    with col1:
        # Daily cluster distribution
        daily_dist = pd.crosstab(df_analysis["day_name"], df_analysis["cluster"])
        # Reorder columns to match sorted clusters
        daily_dist = daily_dist.reindex(columns=sorted_clusters, fill_value=0)
        daily_dist_norm = daily_dist.div(daily_dist.sum(axis=1), axis=0)

        fig = px.bar(
            daily_dist_norm,
            title="Normalized Cluster Distribution by Day of Week",
            labels={"value": "Proportion", "index": "Day of Week"},
            color_discrete_sequence=color_sequence,
        )
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Hourly cluster distribution
        hourly_dist = pd.crosstab(df_analysis["hour"], df_analysis["cluster"])
        # Reorder columns to match sorted clusters
        hourly_dist = hourly_dist.reindex(columns=sorted_clusters, fill_value=0)
        hourly_dist_norm = hourly_dist.div(hourly_dist.sum(axis=1), axis=0)

        fig = px.line(
            hourly_dist_norm,
            title="Cluster Probability by Hour",
            labels={"value": "Probability", "index": "Hour of Day"},
            color_discrete_sequence=color_sequence,
        )
        fig.update_layout(
            height=400, xaxis_title="Hour of Day", yaxis_title="Probability"
        )
        st.plotly_chart(fig, use_container_width=True)

    # Weekend vs Weekday comparison (only if weekends are included)
    if not params["exclude_weekends"]:
        df_analysis["is_weekend"] = df_analysis["day_of_week"].isin([5, 6])
        weekend_dist = pd.crosstab(df_analysis["is_weekend"], df_analysis["cluster"])

        if len(weekend_dist.index) > 1:
            weekend_dist.index = ["Weekday", "Weekend"]
            # Reorder columns to match sorted clusters
            weekend_dist = weekend_dist.reindex(columns=sorted_clusters, fill_value=0)
            weekend_dist_norm = weekend_dist.div(weekend_dist.sum(axis=1), axis=0)

            fig = px.bar(
                weekend_dist_norm,
                title="Cluster Distribution: Weekday vs Weekend",
                labels={"value": "Proportion", "index": "Day Type"},
                color_discrete_sequence=color_sequence,
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            day_type = "Weekend Only" if weekend_dist.index[0] else "Weekday Only"
            st.info(
                f"ℹ️ Analysis period contains {day_type} - no weekend vs weekday comparison available"
            )


def create_summary_analysis(df_pca, df_temporal, params):
    """Create summary analysis."""

    st.subheader("📋 Analysis Summary")

    # Prepare data for analysis
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    df_analysis = df_pca.copy()
    timestamps = pd.to_datetime(df_analysis.index)
    df_analysis["day_of_week"] = timestamps.dayofweek
    df_analysis["day_name"] = df_analysis["day_of_week"].map(lambda x: day_names[x])
    df_analysis["hour"] = timestamps.hour
    df_analysis["date"] = timestamps.date

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Data Points", len(df_analysis))

    with col2:
        st.metric("Unique Clusters", df_analysis["cluster"].nunique())

    with col3:
        st.metric("Days Analyzed", df_analysis["date"].nunique())

    with col4:
        date_range = f"{df_analysis.index.min().strftime('%Y-%m-%d')} to {df_analysis.index.max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)

    # Detailed statistics
    st.subheader("📊 Detailed Statistics")

    # Weekend vs Weekday breakdown
    if not params["exclude_weekends"]:
        df_analysis["is_weekend"] = df_analysis["day_of_week"].isin([5, 6])
        weekday_count = sum(~df_analysis["is_weekend"])
        weekend_count = sum(df_analysis["is_weekend"])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Weekday Data Points", weekday_count)
        with col2:
            st.metric("Weekend Data Points", weekend_count)

    # Cluster distribution by day of week
    st.subheader("📅 Cluster Distribution by Day of Week")
    daily_summary = pd.crosstab(df_analysis["day_name"], df_analysis["cluster"])
    st.dataframe(daily_summary, use_container_width=True)

    # Weekend vs Weekday cluster distribution
    if not params["exclude_weekends"]:
        weekend_count = sum(df_analysis["is_weekend"])
        if weekend_count > 0:
            st.subheader("🗓️ Weekend vs Weekday Cluster Distribution")
            weekend_summary = pd.crosstab(
                df_analysis["is_weekend"], df_analysis["cluster"]
            )
            if len(weekend_summary.index) > 1:
                weekend_summary.index = ["Weekday", "Weekend"]
            else:
                day_type = "Weekend" if weekend_summary.index[0] else "Weekday"
                weekend_summary.index = [day_type]
            st.dataframe(weekend_summary, use_container_width=True)

    # Analysis parameters summary
    st.subheader("⚙️ Analysis Parameters")
    params_df = pd.DataFrame(
        [
            {"Parameter": "Start Date", "Value": params["start_date"]},
            {"Parameter": "Number of Days", "Value": params["num_days"]},
            {"Parameter": "Number of Clusters", "Value": params["clusters"]},
            {
                "Parameter": "Working Hours Only",
                "Value": "Yes" if params["working_hours_only"] else "No",
            },
            {
                "Parameter": "Exclude Weekends",
                "Value": "Yes" if params["exclude_weekends"] else "No",
            },
        ]
    )
    st.dataframe(params_df, use_container_width=True, hide_index=True)


def analyze_cluster_characteristics_detailed(
    df_pca,
    df_scaled,
    df_temporal,
    kmeans,
    feature_names=None,
    cluster_color_map=None,
    cluster_names=None,
):
    """
    Comprehensive analysis of cluster characteristics including temporal patterns,
    statistical summaries, and sensor value distributions.
    """

    st.subheader("🔬 Detailed Cluster Analysis")

    # Get unique clusters
    unique_clusters = sorted(df_pca["cluster"].unique())
    n_clusters = len(unique_clusters)

    # Initialize results dictionary
    cluster_stats = {}

    # Get feature names (exclude temporal features added during clustering)
    if feature_names is None:
        sensor_features = [
            col
            for col in df_scaled.columns
            if col not in ["hour", "day_of_week", "is_weekend", "day_of_period"]
        ]
        feature_names = sensor_features

    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Clusters Analyzed", n_clusters)
    with col2:
        st.metric("Sensor Features", len(feature_names))
    with col3:
        st.metric("Total Data Points", f"{len(df_pca):,}")
    with col4:
        period = f"{df_pca.index.min().strftime('%Y-%m-%d')} to {df_pca.index.max().strftime('%Y-%m-%d')}"
        st.metric("Analysis Period", period)

    # Create expandable sections for each cluster
    for cluster_id in unique_clusters:
        cluster_data = df_pca[df_pca["cluster"] == cluster_id]
        cluster_scaled = df_scaled.loc[cluster_data.index]

        # Get cluster name or use default
        cluster_name = (
            cluster_names.get(cluster_id, f"Cluster {cluster_id}")
            if cluster_names
            else f"Cluster {cluster_id}"
        )

        with st.expander(f"🎯 {cluster_name} - Detailed Analysis", expanded=False):

            # Basic statistics
            n_points = len(cluster_data)
            percentage = (n_points / len(df_pca)) * 100

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", f"{n_points:,}")
            with col2:
                st.metric("Percentage", f"{percentage:.1f}%")
            with col3:
                # Time range
                start_time = cluster_data.index.min()
                end_time = cluster_data.index.max()
                duration = end_time - start_time
                st.metric("Duration", str(duration))

            # Temporal characteristics
            st.markdown("**⏰ Temporal Characteristics**")

            # Hour distribution
            hours = pd.Series(cluster_data.index.hour)
            most_common_hours = hours.value_counts().head(3)
            hour_text = ", ".join(
                [f"{h}:00 ({c} times)" for h, c in most_common_hours.items()]
            )
            st.write(f"• **Most active hours:** {hour_text}")

            # Day of week distribution
            days = pd.Series(cluster_data.index.dayofweek)
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            day_counts = days.value_counts().sort_index()
            day_distribution = ", ".join(
                [
                    f"{day_names[d]}: {count}"
                    for d, count in day_counts.items()
                    if count > 0
                ]
            )
            st.write(f"• **Day distribution:** {day_distribution}")

            # Weekend vs weekday
            is_weekend = days.isin([5, 6])  # Saturday=5, Sunday=6
            weekday_count = sum(~is_weekend)
            weekend_count = sum(is_weekend)
            st.write(
                f"• **Weekday points:** {weekday_count}, **Weekend points:** {weekend_count}"
            )

            # Sensor characteristics
            st.markdown("**🌡️ Sensor Characteristics**")

            sensor_stats = {}
            sensor_data = []

            for feature in feature_names:
                if feature in cluster_scaled.columns:
                    values = cluster_scaled[feature]

                    stats = {
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "median": values.median(),
                        "q25": values.quantile(0.25),
                        "q75": values.quantile(0.75),
                    }

                    sensor_stats[feature] = stats

                    sensor_data.append(
                        {
                            "Feature": feature,
                            "Mean": f"{stats['mean']:.3f}",
                            "Std": f"{stats['std']:.3f}",
                            "Min": f"{stats['min']:.3f}",
                            "Max": f"{stats['max']:.3f}",
                            "Median": f"{stats['median']:.3f}",
                        }
                    )

            if sensor_data:
                sensor_df = pd.DataFrame(sensor_data)
                st.dataframe(sensor_df, use_container_width=True, hide_index=True)

            # Cluster distinctiveness (compared to overall mean)
            st.markdown("**🎯 Cluster Distinctiveness**")
            overall_means = df_scaled[feature_names].mean()
            overall_stds = df_scaled[feature_names].std()

            distinctive_features = []
            for feature in feature_names:
                if feature in cluster_scaled.columns:
                    cluster_mean = sensor_stats[feature]["mean"]
                    z_score = (cluster_mean - overall_means[feature]) / overall_stds[
                        feature
                    ]

                    if abs(z_score) > 1.0:  # More than 1 standard deviation away
                        direction = "↑ HIGH" if z_score > 0 else "↓ LOW"
                        distinctive_features.append(
                            f"**{feature}:** {direction} ({z_score:+.2f}σ)"
                        )

            if distinctive_features:
                st.write("Notable deviations from overall average:")
                for feature in distinctive_features[:5]:  # Show top 5
                    st.write(f"• {feature}")
            else:
                st.write("• No significant deviations from overall average")

            # Store results
            cluster_stats[cluster_id] = {
                "name": cluster_name,
                "basic_stats": {
                    "n_points": n_points,
                    "percentage": percentage,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                },
                "temporal_stats": {
                    "most_common_hours": most_common_hours.to_dict(),
                    "day_distribution": day_counts.to_dict(),
                    "weekday_count": weekday_count,
                    "weekend_count": weekend_count,
                },
                "sensor_stats": sensor_stats,
                "distinctive_features": distinctive_features,
            }

    # Create comprehensive visualizations
    create_cluster_visualizations(
        cluster_stats, df_pca, df_scaled, feature_names, cluster_color_map
    )

    return cluster_stats


def create_cluster_visualizations(
    cluster_stats, df_pca, df_scaled, feature_names, cluster_color_map
):
    """Create comprehensive visualizations of cluster characteristics using Plotly."""

    st.subheader("📊 Cluster Characteristics Visualizations")

    cluster_ids = sorted(list(cluster_stats.keys()))  # Ensure sorted order
    n_clusters = len(cluster_ids)

    # 1. Cluster sizes and durations
    col1, col2 = st.columns(2)

    with col1:
        # Cluster sizes
        sizes = [cluster_stats[cid]["basic_stats"]["n_points"] for cid in cluster_ids]
        cluster_labels = [f"Cluster {cid}" for cid in cluster_ids]

        fig = px.bar(
            x=cluster_labels,
            y=sizes,
            title="Cluster Sizes",
            labels={"x": "Cluster", "y": "Number of Data Points"},
            color=cluster_labels,
            color_discrete_map={
                f"Cluster {cid}": cluster_color_map[str(cid)] for cid in cluster_ids
            },
        )

        # Add percentage labels
        total_points = sum(sizes)
        for i, (size, label) in enumerate(zip(sizes, cluster_labels)):
            fig.add_annotation(
                x=i,
                y=size,
                text=f"{size:,}<br>({size/total_points*100:.1f}%)",
                showarrow=False,
                yshift=10,
            )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Cluster durations
        durations = []
        for cluster_id in cluster_ids:
            cluster_data = df_pca[df_pca["cluster"] == cluster_id]
            if len(cluster_data) > 1:
                duration = (
                    cluster_data.index.max() - cluster_data.index.min()
                ).total_seconds() / 3600  # hours
            else:
                duration = 0
            durations.append(duration)

        fig = px.bar(
            x=cluster_labels,
            y=durations,
            title="Cluster Duration (Hours)",
            labels={"x": "Cluster", "y": "Duration (Hours)"},
            color=cluster_labels,
            color_discrete_map={
                f"Cluster {cid}": cluster_color_map[str(cid)] for cid in cluster_ids
            },
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2. Temporal patterns
    st.markdown("**⏰ Temporal Pattern Analysis**")

    col1, col2 = st.columns(2)

    with col1:
        # Hour distribution heatmap
        hour_data = []
        for cluster_id in cluster_ids:
            cluster_data = df_pca[df_pca["cluster"] == cluster_id]
            hour_counts = pd.Series(cluster_data.index.hour).value_counts()

            for hour in range(24):
                count = hour_counts.get(hour, 0)
                hour_data.append(
                    {"Cluster": f"Cluster {cluster_id}", "Hour": hour, "Count": count}
                )

        hour_df = pd.DataFrame(hour_data)

        # Create pivot table for heatmap
        hour_pivot = hour_df.pivot(
            index="Cluster", columns="Hour", values="Count"
        ).fillna(0)

        fig = px.imshow(
            hour_pivot.values,
            x=hour_pivot.columns,
            y=hour_pivot.index,
            title="Hourly Activity Patterns",
            labels={"x": "Hour of Day", "y": "Cluster", "color": "Activity Count"},
            color_continuous_scale="viridis",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Weekend vs Weekday distribution
        weekday_counts = [
            cluster_stats[cid]["temporal_stats"]["weekday_count"] for cid in cluster_ids
        ]
        weekend_counts = [
            cluster_stats[cid]["temporal_stats"]["weekend_count"] for cid in cluster_ids
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                name="Weekday",
                x=cluster_labels,
                y=weekday_counts,
                marker_color=TAB10_COLORS[0],
            )
        )
        fig.add_trace(
            go.Bar(
                name="Weekend",
                x=cluster_labels,
                y=weekend_counts,
                marker_color=TAB10_COLORS[1],
            )
        )

        fig.update_layout(
            title="Weekday vs Weekend Distribution",
            xaxis_title="Cluster",
            yaxis_title="Number of Data Points",
            barmode="group",
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3. Feature analysis
    st.markdown("**🌡️ Sensor Feature Analysis**")

    # Calculate feature means for each cluster (using sorted order)
    feature_data = []
    for cluster_id in sorted(cluster_ids):  # Ensure sorted order
        cluster_data = df_pca[df_pca["cluster"] == cluster_id]
        cluster_scaled = df_scaled.loc[cluster_data.index]

        for feature in feature_names:
            if feature in cluster_scaled.columns:
                mean_val = cluster_scaled[feature].mean()
                feature_data.append(
                    {
                        "Cluster": f"Cluster {cluster_id}",
                        "Feature": feature,
                        "Mean_Value": mean_val,
                    }
                )

    feature_df = pd.DataFrame(feature_data)

    # Feature heatmap
    feature_pivot = feature_df.pivot(
        index="Cluster", columns="Feature", values="Mean_Value"
    )

    fig = px.imshow(
        feature_pivot.values,
        x=feature_pivot.columns,
        y=feature_pivot.index,
        title="Feature Means by Cluster (Scaled Values)",
        labels={"x": "Features", "y": "Cluster", "color": "Scaled Value"},
        color_continuous_scale="RdBu_r",
    )
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    # Top variable features
    feature_variability = {}
    for feature in feature_names:
        feature_values = feature_df[feature_df["Feature"] == feature][
            "Mean_Value"
        ].values
        feature_variability[feature] = np.std(feature_values)

    top_features = sorted(
        feature_variability.items(), key=lambda x: x[1], reverse=True
    )[:4]

    if len(top_features) >= 2:
        col1, col2 = st.columns(2)

        for idx, (feature, _) in enumerate(top_features[:2]):
            with col1 if idx == 0 else col2:
                feature_subset = feature_df[feature_df["Feature"] == feature]

                fig = px.bar(
                    feature_subset,
                    x="Cluster",
                    y="Mean_Value",
                    title=f"{feature} by Cluster",
                    color="Cluster",
                    color_discrete_map={
                        f"Cluster {cid}": cluster_color_map[str(cid)]
                        for cid in cluster_ids
                    },
                    category_orders={
                        "Cluster": [f"Cluster {cid}" for cid in sorted(cluster_ids)]
                    },
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)


def display_realtime_prediction_interface(kmeans, sensor_scaler, feature_names, params):
    """Display real-time cluster prediction interface for new data points."""
    
    # Only show if we have the required components
    if not all([kmeans is not None, sensor_scaler is not None, feature_names is not None and len(feature_names) > 0]):
        return
    
    st.divider()
    st.header("🔮 Real-time Cluster Prediction")
    st.markdown("Input new sensor data points to see which cluster they match closest to.")
    
    # Get cluster centroids and transform them back to original scale
    cluster_centroids = kmeans.cluster_centers_
    n_clusters = len(cluster_centroids)
    
    # Transform centroids back to original scale for display
    centroids_original = sensor_scaler.inverse_transform(cluster_centroids)
    
    # Create dataframe with cluster centroids
    centroids_df = pd.DataFrame(centroids_original, columns=feature_names)
    centroids_df.index = [f"Cluster {i}" for i in range(n_clusters)]
    
    # Get cluster sizes from stored results to find most popular cluster
    most_popular_cluster = 0
    if "analysis_results" in st.session_state:
        df_pca = st.session_state.analysis_results["df_pca"]
        cluster_counts = df_pca["cluster"].value_counts().sort_index()
        most_popular_cluster = int(cluster_counts.idxmax())
    
    # Get min/max values from the original data for realistic bounds
    # Use the analysis results to get the original data bounds
    data_bounds = {}
    if "analysis_results" in st.session_state:
        # Get original data bounds from the scaled data by inverse transforming extreme values
        df_scaled = st.session_state.analysis_results["df_scaled"]
        
        for i, feature in enumerate(feature_names):
            if feature in df_scaled.columns:
                # Get min/max of scaled data
                scaled_min = df_scaled[feature].min()
                scaled_max = df_scaled[feature].max()
                
                # Create dummy arrays to inverse transform
                dummy_min = np.zeros(len(feature_names))
                dummy_max = np.zeros(len(feature_names))
                dummy_min[i] = scaled_min
                dummy_max[i] = scaled_max
                
                # Inverse transform to get original bounds
                original_min = sensor_scaler.inverse_transform([dummy_min])[0][i]
                original_max = sensor_scaler.inverse_transform([dummy_max])[0][i]
                
                data_bounds[feature] = {
                    "min": float(original_min),
                    "max": float(original_max),
                    "range": float(original_max - original_min)
                }
    
    # Fallback bounds if we can't get from data
    fallback_bounds = {
        "gas_resistance": {"min": 0.0, "max": 10000000.0, "range": 10000000.0},
        "pressure": {"min": 900.0, "max": 1100.0, "range": 200.0},
        "bme_temp": {"min": -10.0, "max": 50.0, "range": 60.0},
        "bme_hum": {"min": 0.0, "max": 100.0, "range": 100.0},
        "co2": {"min": 300.0, "max": 5000.0, "range": 4700.0},
        "mass_conc_pm1": {"min": 0.0, "max": 1000.0, "range": 1000.0},
        "mass_conc_pm2_5": {"min": 0.0, "max": 1000.0, "range": 1000.0},
        "mass_conc_pm4": {"min": 0.0, "max": 1000.0, "range": 1000.0},
        "mass_conc_pm10": {"min": 0.0, "max": 1000.0, "range": 1000.0},
        "num_conc_pm0_5": {"min": 0.0, "max": 10000.0, "range": 10000.0},
        "num_conc_pm1": {"min": 0.0, "max": 10000.0, "range": 10000.0},
        "num_conc_pm2_5": {"min": 0.0, "max": 10000.0, "range": 10000.0},
        "num_conc_pm4": {"min": 0.0, "max": 10000.0, "range": 10000.0},
        "num_conc_pm10": {"min": 0.0, "max": 10000.0, "range": 10000.0},
        "particle_size": {"min": 0.0, "max": 10.0, "range": 10.0},
    }
    
    # Create expandable section to show cluster centroids
    with st.expander("📊 View Cluster Centroids (Original Scale)", expanded=False):
        st.markdown("**Cluster centroids in original sensor units:**")
        st.dataframe(centroids_df, use_container_width=True)
        st.info(f"💡 Most popular cluster: **Cluster {most_popular_cluster}** (used as default values below)")
    
    # Create input form
    with st.form("prediction_form"):
        st.subheader("📊 Enter Sensor Values")
        
        # Add option to select starting values
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Choose starting values:**")
            
        with col2:
            preset_option = st.selectbox(
                "Preset Values",
                options=[f"Cluster {i}" for i in range(n_clusters)] + ["Custom"],
                index=most_popular_cluster,
                help="Select cluster centroid as starting values or use custom values"
            )
        
        # Get default values based on selection
        if preset_option != "Custom":
            cluster_idx = int(preset_option.split()[1])
            default_values = {
                feature: centroids_original[cluster_idx][i] 
                for i, feature in enumerate(feature_names)
            }
            st.info(f"📋 Using **{preset_option}** centroid values as starting points")
        else:
            # Use overall reasonable defaults for custom
            default_values = {
                feature: centroids_original[most_popular_cluster][i] 
                for i, feature in enumerate(feature_names)
            }
            st.info(f"📋 Using **Cluster {most_popular_cluster}** centroid values as starting points")
        
        # Create input fields for each sensor feature
        input_values = {}
        
        # Generate sensor configs based on actual data bounds
        def get_sensor_config(feature):
            """Get sensor configuration based on actual data bounds."""
            bounds = data_bounds.get(feature, fallback_bounds.get(feature, {
                "min": 0.0, "max": 1000.0, "range": 1000.0
            }))
            
            # Calculate appropriate step size based on range
            range_val = bounds["range"]
            if range_val > 10000:
                step = 1000.0  # Ensure float type
                format_str = "%.0f"
            elif range_val > 1000:
                step = 100.0  # Ensure float type
                format_str = "%.0f"
            elif range_val > 100:
                step = 10.0   # Ensure float type
                format_str = "%.0f"
            elif range_val > 10:
                step = 1.0    # Ensure float type
                format_str = "%.0f"
            elif range_val > 1:
                step = 0.1
                format_str = "%.1f"
            else:
                step = 0.01
                format_str = "%.2f"
            
            return {
                "min": float(bounds["min"]),  # Ensure float type
                "max": float(bounds["max"]),  # Ensure float type
                "step": step,
                "format": format_str
            }
        
        # Create columns for organized input
        n_cols = 3
        cols = st.columns(n_cols)
        col_idx = 0
        
        for feature in feature_names:
            with cols[col_idx % n_cols]:
                config = get_sensor_config(feature)
                
                # Get default value from selected preset
                default_value = default_values.get(feature, 0.0)
                
                # Clean up feature name for display
                display_name = feature.replace("_", " ").title()
                
                input_values[feature] = st.number_input(
                    display_name,
                    min_value=config["min"],
                    max_value=config["max"],
                    value=float(default_value),
                    step=config["step"],
                    format=config["format"],
                    key=f"input_{feature}_{preset_option}",  # Include preset in key to force update
                    help=f"Default from {preset_option}: {default_value:.3f} | Range: {config['min']:.2f} - {config['max']:.2f}"
                )
            col_idx += 1
        
        # Predict button
        predict_button = st.form_submit_button("🔮 Predict Cluster", type="primary")
        
        if predict_button:
            try:
                # Prepare the input data
                new_data = []
                
                # Sensor features
                for feature in feature_names:
                    new_data.append(input_values[feature])
                
                # Create DataFrame for preprocessing
                new_df = pd.DataFrame([new_data], columns=feature_names)
                
                # Apply sensor scaling
                new_df_scaled = pd.DataFrame(
                    sensor_scaler.transform(new_df), 
                    columns=feature_names
                )
                
                # Use only sensor features for prediction
                all_features = new_df_scaled
                
                # Predict cluster
                predicted_cluster = kmeans.predict(all_features)[0]
                
                # Calculate distances to all cluster centers
                distances = kmeans.transform(all_features)[0]
                
                # Display results
                st.success(f"✅ Prediction Complete!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Cluster", str(predicted_cluster))
                
                with col2:
                    confidence = 1 - (distances[predicted_cluster] / distances.max())
                    st.metric("Confidence", f"{confidence:.1%}")
                
                with col3:
                    st.metric("Distance to Center", f"{distances[predicted_cluster]:.2f}")
                
                # Show distances to all clusters
                st.subheader("📊 Distance to All Clusters")
                
                distance_data = []
                for i, distance in enumerate(distances):
                    is_predicted = i == predicted_cluster
                    distance_data.append({
                        "Cluster": str(i),
                        "Distance": distance,
                        "Predicted": "✅ YES" if is_predicted else "No"
                    })
                
                distance_df = pd.DataFrame(distance_data)
                distance_df = distance_df.sort_values("Distance")
                
                # Style the dataframe
                def highlight_predicted(row):
                    if row["Predicted"] == "✅ YES":
                        return ["background-color: #90EE90"] * len(row)
                    return [""] * len(row)
                
                st.dataframe(
                    distance_df.style.apply(highlight_predicted, axis=1),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Show input summary
                st.subheader("📋 Input Summary")
                
                input_summary = []
                
                # Sensor inputs
                for feature in feature_names:
                    bounds = data_bounds.get(feature, fallback_bounds.get(feature, {"min": 0, "max": 1000}))
                    input_summary.append({
                        "Feature": feature.replace("_", " ").title(),
                        "Input Value": input_values[feature],
                        "Scaled Value": new_df_scaled[feature].iloc[0],
                        "Data Range": f"{bounds['min']:.2f} - {bounds['max']:.2f}"
                    })
                
                input_summary_df = pd.DataFrame(input_summary)
                st.dataframe(input_summary_df, use_container_width=True, hide_index=True)
                
                # Tips for interpretation
                st.info("""
                💡 **How to interpret the results:**
                - **Predicted Cluster**: The cluster that best matches your input data
                - **Confidence**: How certain the model is (based on distance to cluster centers)
                - **Distance**: Lower distances indicate better matches
                - **Scaled Values**: Your inputs after applying the same normalization used during training
                - **Data Range**: The actual min/max values seen in the training data
                """)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                st.exception(e)
        

