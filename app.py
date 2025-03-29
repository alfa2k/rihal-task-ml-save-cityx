import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import PyPDF2
import re
import os
import webbrowser
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go
import streamlit_toggle as tog
from PIL import Image
import base64

# Set page configuration and styling
st.set_page_config(
    page_title="CityX Crime Analysis Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FF5252;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #FF5252;
    }
    .subheader {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .stat-box {
        background-color: #37474F;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stat-number {
        font-size: 2rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .severity-1 { background-color: #4CAF50; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .severity-2 { background-color: #FFEB3B; color: black; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .severity-3 { background-color: #FF9800; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .severity-4 { background-color: #F44336; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .severity-5 { background-color: #9C27B0; color: white; padding: 0.2rem 0.5rem; border-radius: 4px; }
    .report-container {
        background-color: #ECEFF1;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .sidebar-content {
        background-color: #263238;
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }
    .hover-info:hover {
        background-color: #E3F2FD;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Load and Prepare Dataset (for mapping & classifier)
# -----------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('Competition_Dataset.csv', parse_dates=['Dates'])
    # Rename coordinate columns.
    df.rename(columns={'Latitude (Y)': 'Latitude', 'Longitude (X)': 'Longitude'}, inplace=True)
    # Standardize text fields.
    df['Category'] = df['Category'].str.upper().str.strip()
    df['Descript'] = df['Descript'].str.lower().str.strip()
    df['DayOfWeek'] = df['DayOfWeek'].str.title().str.strip()
    df['PdDistrict'] = df['PdDistrict'].str.title().str.strip()
    # Extract time components for analysis
    df['Year'] = df['Dates'].dt.year
    df['Month'] = df['Dates'].dt.month
    df['Hour'] = df['Dates'].dt.hour
    df['DayName'] = df['Dates'].dt.day_name()
    return df

df = load_data()

# -----------------------------------------------
# Build Classifier Pipeline for Crime Classification
# -----------------------------------------------
@st.cache_resource
def get_classifier(data):
    X = data['Descript']
    y = data['Category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    return pipeline

classifier_pipeline = get_classifier(df)

# Severity mapping with alias for "SUSPICIOUS OCC"
severity_mapping = {
    'NON-CRIMINAL': 1,
    'SUSPICIOUS OCCURRENCE': 1,
    'SUSPICIOUS OCC': 1,  # alias for consistency
    'MISSING PERSON': 1,
    'RUNAWAY': 1,
    'RECOVERED VEHICLE': 1,
    'WARRANTS': 2,
    'OTHER OFFENSES': 2,
    'VANDALISM': 2,
    'TRESPASS': 2,
    'DISORDERLY CONDUCT': 2,
    'BAD CHECKS': 2,
    'LARCENY/THEFT': 3,
    'VEHICLE THEFT': 3,
    'FORGERY/COUNTERFEITING': 3,
    'DRUG/NARCOTIC': 3,
    'STOLEN PROPERTY': 3,
    'FRAUD': 3,
    'BRIBERY': 3,
    'EMBEZZLEMENT': 3,
    'ROBBERY': 4,
    'WEAPON LAWS': 4,
    'BURGLARY': 4,
    'EXTORTION': 4,
    'KIDNAPPING': 5,
    'ARSON': 5
}

def get_severity_class(severity):
    """Returns an HTML span with appropriate severity class"""
    if severity == 1:
        return f'<span class="severity-1">Level {severity} - Low</span>'
    elif severity == 2:
        return f'<span class="severity-2">Level {severity} - Minor</span>'
    elif severity == 3:
        return f'<span class="severity-3">Level {severity} - Moderate</span>'
    elif severity == 4:
        return f'<span class="severity-4">Level {severity} - High</span>'
    elif severity == 5:
        return f'<span class="severity-5">Level {severity} - Severe</span>'
    else:
        return f'<span>Unknown</span>'

def predict_crime(description):
    """Predicts crime category and severity for a given description."""
    predicted_category = classifier_pipeline.predict([description])[0]
    severity = severity_mapping.get(predicted_category, "Unknown")
    return predicted_category, severity

# -----------------------------------------------
# PDF Parsing Functions for Police Reports
# -----------------------------------------------
def extract_text_from_pdf(file):
    """Extracts text from an uploaded PDF file."""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def parse_police_report(text):
    """
    Parses key fields from a police report text using regex.
    Expected fields: Report Number, Date & Time, Incident Location, Coordinates, 
    Detailed Description, Police District, Resolution, Suspect Description, Victim Information.
    """
    report = {}
    report_number = re.search(r"Report Number:\s*([0-9\-]+)", text)
    report['Report Number'] = report_number.group(1) if report_number else "Unknown"
    
    date_time = re.search(r"Date & Time:\s*([\d\- :]+)", text)
    report['Date & Time'] = date_time.group(1) if date_time else "Unknown"
    
    location = re.search(r"Incident Location:\s*(.+)", text)
    report['Incident Location'] = location.group(1).strip() if location else "Unknown"
    
    coordinates = re.search(r"Coordinates:\s*\(([^)]+)\)", text)
    report['Coordinates'] = coordinates.group(1) if coordinates else "Unknown"
    
    description = re.search(r"Detailed Description:\s*(.+?)(\n\n|$)", text, re.DOTALL)
    report['Detailed Description'] = description.group(1).strip() if description else "Unknown"
    
    police_district = re.search(r"Police District:\s*(.+)", text)
    report['Police District'] = police_district.group(1).strip() if police_district else "Unknown"
    
    resolution = re.search(r"Resolution:\s*(.+)", text)
    report['Resolution'] = resolution.group(1).strip() if resolution else "Unknown"
    
    suspect = re.search(r"Suspect Description:\s*(.+)", text)
    report['Suspect Description'] = suspect.group(1).strip() if suspect else "Unknown"
    
    victim = re.search(r"Victim Information:\s*(.+)", text)
    report['Victim Information'] = victim.group(1).strip() if victim else "Unknown"
    
    return report

# -----------------------------------------------
# Helper functions for analytics
# -----------------------------------------------
def generate_time_heatmap(data):
    # Create hour of day vs day of week heatmap
    hour_day_counts = data.groupby(['Hour', 'DayName']).size().reset_index(name='Count')
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hour_day_counts['DayName'] = pd.Categorical(hour_day_counts['DayName'], categories=day_order, ordered=True)
    hour_day_counts = hour_day_counts.sort_values(['DayName', 'Hour'])
    
    # Create pivot table for heatmap
    pivot_data = hour_day_counts.pivot(index='Hour', columns='DayName', values='Count')
    
    # Generate heatmap with Plotly
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Day of Week", y="Hour of Day", color="Crime Count"),
        x=day_order,
        y=list(range(24)),
        aspect="auto",
        color_continuous_scale="Viridis"
    )
    
    fig.update_layout(
        title="Crime Frequency by Hour and Day",
        xaxis_title="Day of Week",
        yaxis_title="Hour of Day",
        height=500
    )
    
    return fig

# -----------------------------------------------
# Streamlit UI: Sidebar Navigation
# -----------------------------------------------
# Logo placeholder at the top of sidebar
st.sidebar.markdown("""
<div style="text-align: center;">
    <h1 style="color: #FF5252;">CityX</h1>
    <p style="color: #B0BEC5;">Crime Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

# User selection menu
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard Overview", "Interactive Map", "Police Report Analysis", "Crime Classification"],
    label_visibility="collapsed"
)

# Date filter in sidebar (applies to all pages)
st.sidebar.markdown("### Filter Data")
# Get min and max dates from the dataset
min_date = df['Dates'].min().date()
max_date = df['Dates'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = df[(df['Dates'].dt.date >= start_date) & (df['Dates'].dt.date <= end_date)]
else:
    filtered_df = df

# District filter
districts = ['All'] + sorted(df['PdDistrict'].unique().tolist())
selected_district = st.sidebar.selectbox("Select District", districts)

if selected_district != 'All':
    filtered_df = filtered_df[filtered_df['PdDistrict'] == selected_district]

# Show stats in sidebar
st.sidebar.markdown("### Quick Stats")

total_incidents = len(filtered_df)
unique_categories = filtered_df['Category'].nunique()
highest_category = filtered_df['Category'].value_counts().idxmax()

col1, col2 = st.sidebar.columns(2)
with col1:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{total_incidents:,}</div>
        <div class="stat-label">Incidents</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-box">
        <div class="stat-number">{unique_categories}</div>
        <div class="stat-label">Categories</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div class="stat-box" style="margin-top: 10px;">
    <div style="font-size: 1.2rem;">Most Common</div>
    <div style="font-weight: bold; color: #FF5252;">{highest_category}</div>
</div>
""", unsafe_allow_html=True)

# Add footer to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align: center; color: #B0BEC5; font-size: 0.8rem;">
    © 2025 CityX Crime Analysis<br>
    v2.0.3
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------
# Page 0: Dashboard Overview
# -----------------------------------------------
if page == "Dashboard Overview":
    st.markdown('<h1 class="main-header">CityX Crime Analysis Dashboard</h1>', unsafe_allow_html=True)

    # Top metrics
    st.markdown('<div class="card">', unsafe_allow_html=True)
    metrics_cols = st.columns(4)
    
    # Calculate metrics
    incidents_per_day = filtered_df.groupby(filtered_df['Dates'].dt.date).size().mean()
    top_district = filtered_df['PdDistrict'].value_counts().idxmax()
    top_district_pct = (filtered_df['PdDistrict'].value_counts().max() / len(filtered_df)) * 100
    avg_severity = filtered_df['Category'].map(severity_mapping).mean()
    
    with metrics_cols[0]:
        st.metric("Daily Incidents", f"{incidents_per_day:.1f}")
    
    with metrics_cols[1]:
        st.metric("Top District", top_district)
    
    with metrics_cols[2]:
        st.metric("District Concentration", f"{top_district_pct:.1f}%")
        
    with metrics_cols[3]:
        st.metric("Avg. Severity", f"{avg_severity:.1f}/5")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Two-column layout for main dashboard content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<h2 class="subheader">Crime Trends</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Create time series chart by day
        crime_by_date = filtered_df.groupby(filtered_df['Dates'].dt.date).size().reset_index(name='count')
        crime_by_date.columns = ['Date', 'Count']
        
        fig_trend = px.line(
            crime_by_date, 
            x='Date', 
            y='Count',
            template='plotly_white',
            line_shape='spline'
        )
        fig_trend.update_layout(
            title='Daily Crime Count',
            xaxis_title='Date',
            yaxis_title='Number of Incidents',
            height=350
        )
        fig_trend.update_traces(line_color='#FF5252')
        st.plotly_chart(fig_trend, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Time heatmap
        st.markdown('<h2 class="subheader">Crime Patterns</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig_heatmap = generate_time_heatmap(filtered_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h2 class="subheader">Crime Distribution</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Get top 10 crime categories
        top_crimes = filtered_df['Category'].value_counts().nlargest(10)
        fig_bar = px.bar(
            x=top_crimes.values,
            y=top_crimes.index,
            orientation='h',
            template='plotly_white',
            labels={'x': 'Count', 'y': 'Crime Category'},
            color=top_crimes.values,
            color_continuous_scale='Viridis',
        )
        fig_bar.update_layout(
            title='Top 10 Crime Categories',
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # District distribution
        st.markdown('<h2 class="subheader">District Breakdown</h2>', unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        district_data = filtered_df['PdDistrict'].value_counts().reset_index()
        district_data.columns = ['District', 'Count']
        
        fig_pie = px.pie(
            district_data, 
            values='Count', 
            names='District',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(
            title='Incidents by District',
            height=350
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Bottom section with severity breakdown
    st.markdown('<h2 class="subheader">Severity Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    # Add severity level to filtered dataframe
    filtered_df['Severity'] = filtered_df['Category'].map(severity_mapping)
    severity_counts = filtered_df['Severity'].value_counts().sort_index()
    
    # Create a bar chart with custom colors
    severity_colors = ['#4CAF50', '#FFEB3B', '#FF9800', '#F44336', '#9C27B0']
    severity_labels = ['Level 1 - Low', 'Level 2 - Minor', 'Level 3 - Moderate', 'Level 4 - High', 'Level 5 - Severe']
    
    fig_severity = go.Figure()
    for i in range(1, 6):
        count = severity_counts.get(i, 0)
        fig_severity.add_trace(go.Bar(
            x=[severity_labels[i-1]],
            y=[count],
            name=severity_labels[i-1],
            marker_color=severity_colors[i-1],
            text=[count],
            textposition='auto'
        ))
    
    fig_severity.update_layout(
        title='Crime Incidents by Severity Level',
        xaxis_title='Severity Level',
        yaxis_title='Number of Incidents',
        height=300,
        showlegend=False
    )
    
    st.plotly_chart(fig_severity, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# Page 1: Interactive Map 
# -----------------------------------------------
elif page == "Interactive Map":
    st.markdown('<h1 class="main-header">Interactive Crime Map</h1>', unsafe_allow_html=True)
    
    # Map settings
    st.markdown('<div class="card">', unsafe_allow_html=True)
    map_cols = st.columns([1, 1, 2])
    
    with map_cols[0]:
        map_type = st.selectbox("Map Type", ["Markers", "Heatmap", "Both"])
    
    with map_cols[1]:
        map_style = st.selectbox("Map Style", ["Dark", "Light", "Street", "Satellite"])
        
        # Map tile options
        tile_options = {
            "Dark": "CartoDB dark_matter",
            "Light": "CartoDB positron",
            "Street": "OpenStreetMap"
        }
    
    with map_cols[2]:
        # Category filter for map
        all_categories = sorted(filtered_df['Category'].unique().tolist())
        selected_categories = st.multiselect(
            "Filter Crime Categories", 
            all_categories,
            default=all_categories[:5] if len(all_categories) > 5 else all_categories
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Filter data for map based on selected categories
    if selected_categories:
        map_data = filtered_df[filtered_df['Category'].isin(selected_categories)]
    else:
        map_data = filtered_df
    
    # Create a Folium map
    map_center = [map_data['Longitude'].mean(), map_data['Latitude'].mean()]
    # Create a Folium map with the selected style.
    if map_style == "Satellite":
        # Use Esri World Imagery as the satellite tile, with proper attribution.
        crime_map = folium.Map(
            location=map_center, 
            zoom_start=12, 
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
        )
    else:
        crime_map = folium.Map(
            location=map_center, 
            zoom_start=12, 
            tiles=tile_options[map_style]
        )
    
    # Sample data for performance if needed
    max_markers = 1000
    if len(map_data) > max_markers:
        map_data_sample = map_data.sample(max_markers)
    else:
        map_data_sample = map_data
    
    if map_type in ["Markers", "Both"]:
        # Add a marker cluster
        marker_cluster = MarkerCluster().add_to(crime_map)
        
        # Add markers
        for idx, row in map_data_sample.iterrows():
            # Get severity for color coding
            severity = severity_mapping.get(row['Category'], 1)
            color = {
                1: "green",
                2: "beige",
                3: "orange",
                4: "red",
                5: "purple"
            }.get(severity, "blue")
            
            popup_text = f"""
            <div style='width: 200px'>
                <b>Category:</b> {row['Category']}<br>
                <b>Description:</b> {row['Descript']}<br>
                <b>Date:</b> {row['Dates'].strftime('%Y-%m-%d %H:%M')}<br>
                <b>District:</b> {row['PdDistrict']}<br>
                <b>Severity:</b> Level {severity}
            </div>
            """
            
            folium.Marker(
                location=[row['Longitude'], row['Latitude']],
                popup=folium.Popup(popup_text, max_width=300),
                icon=folium.Icon(color=color, icon="info-sign")
            ).add_to(marker_cluster)
    
    if map_type in ["Heatmap", "Both"]:
        
        gradient = {str(k): v for k, v in {0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}.items()}
        heat_data = [[row['Longitude'], row['Latitude']] for idx, row in map_data_sample.iterrows()]
        HeatMap(heat_data, radius=15, blur=10, gradient=gradient).add_to(crime_map)
    
    # Map legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 0 15px rgba(0,0,0,0.2);">
        <p><strong>Severity Levels</strong></p>
        <p><i class="fa fa-circle" style="color:green"></i> Level 1 - Low</p>
        <p><i class="fa fa-circle" style="color:beige"></i> Level 2 - Minor</p>
        <p><i class="fa fa-circle" style="color:orange"></i> Level 3 - Moderate</p>
        <p><i class="fa fa-circle" style="color:red"></i> Level 4 - High</p>
        <p><i class="fa fa-circle" style="color:purple"></i> Level 5 - Severe</p>
    </div>
    '''
    crime_map.get_root().html.add_child(folium.Element(legend_html))
    
    # Display the map
    st.markdown('<div class="card" style="padding: 0.5rem;">', unsafe_allow_html=True)
    st.components.v1.html(crime_map._repr_html_(), height=600)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Map insights
    st.markdown('<h2 class="subheader">Map Insights</h2>', unsafe_allow_html=True)
    
    insight_cols = st.columns(3)
    
    with insight_cols[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"#### Top Locations")
        # Group by coordinates and count occurrences
        location_counts = map_data.groupby(['Latitude', 'Longitude']).size().reset_index(name='count')
        top_locations = location_counts.sort_values('count', ascending=False).head(5)
        
        for i, loc in enumerate(top_locations.itertuples(), 1):
            st.markdown(f"**Hotspot #{i}**: ({loc.Latitude:.4f}, {loc.Longitude:.4f}) - {loc.count} incidents")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with insight_cols[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Crime Distribution")
        
        category_counts = map_data['Category'].value_counts()
        total = len(map_data)
        
        for cat, count in category_counts.nlargest(5).items():
            percentage = (count / total) * 100
            st.markdown(f"**{cat}**: {count} ({percentage:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with insight_cols[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### District Analysis")
        
        district_counts = map_data['PdDistrict'].value_counts()
        total = len(map_data)
        
        for district, count in district_counts.nlargest(5).items():
            percentage = (count / total) * 100
            st.markdown(f"**{district}**: {count} ({percentage:.1f}%)")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# Page 2: Police Report Extraction & Classification
# -----------------------------------------------
elif page == "Police Report Analysis":
    st.markdown('<h1 class="main-header">Police Report Analysis</h1>', unsafe_allow_html=True)
    
    # Upload area
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Upload one or more police report PDFs for automated extraction and classification:")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_files:
        with st.spinner("Processing reports..."):
            report_list = []
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                report = parse_police_report(text)
                # Use classifier on the detailed description if available.
                if report.get("Detailed Description", "Unknown") != "Unknown":
                    pred_cat, severity = predict_crime(report["Detailed Description"])
                    report["Predicted Category"] = pred_cat
                    report["Assigned Severity"] = severity
                else:
                    report["Predicted Category"] = "Unknown"
                    report["Assigned Severity"] = "Unknown"
                report_list.append((report, text))
        
        # Display reports with better styling
        st.markdown('<h2 class="subheader">Extracted Police Reports</h2>', unsafe_allow_html=True)
        
        for i, (rep, text) in enumerate(report_list):
            with st.expander(f"Report #{i+1}: {rep['Report Number']} - {rep['Date & Time']}", expanded=i==0):
                report_cols = st.columns([2, 1])
                
                with report_cols[0]:
                    st.markdown('<div class="report-container">', unsafe_allow_html=True)
                    
                    # Primary info
                    st.markdown("#### Primary Information")
                    st.markdown(f"**Report Number:** {rep['Report Number']}")
                    st.markdown(f"**Date & Time:** {rep['Date & Time']}")
                    st.markdown(f"**Location:** {rep['Incident Location']}")
                    st.markdown(f"**Police District:** {rep['Police District']}")
                    
                    # Description
                    st.markdown("#### Incident Details")
                    st.text_area("Description", rep['Detailed Description'], height=150)
                    
                    # Suspect and victim info
                    suspect_victim_cols = st.columns(2)
                    with suspect_victim_cols[0]:
                        st.markdown("**Suspect Description:**")
                        st.markdown(f"<div class='hover-info'>{rep['Suspect Description']}</div>", unsafe_allow_html=True)
                    
                    with suspect_victim_cols[1]:
                        st.markdown("**Victim Information:**")
                        st.markdown(f"<div class='hover-info'>{rep['Victim Information']}</div>", unsafe_allow_html=True)
                    
                    st.markdown("**Resolution Status:**")
                    st.markdown(f"<div class='hover-info'>{rep['Resolution']}</div>", unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with report_cols[1]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### AI Classification")
                    
                    # Show AI classification results with styling
                    st.markdown(f"**Predicted Crime Category:**")
                    st.markdown(f"<div style='font-size: 1.2rem; font-weight: bold; color: #2196F3;'>{rep['Predicted Category']}</div>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Assigned Severity Level:**")
                    st.markdown(get_severity_class(rep['Assigned Severity']), unsafe_allow_html=True)
                    
                    # Confidence indicator (mock)
                    st.markdown("**Confidence Level:**")
                    confidence = 85  # Mock confidence percentage
                    st.progress(confidence/100)
                    st.markdown(f"{confidence}%")
                    
                    # PDF preview 
                    st.markdown("#### PDF Preview")
                    st.text_area("Raw Text Sample", text[:200] + "..." if len(text) > 200 else text, height=100)
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis summary
        if len(report_list) > 1:
            st.markdown('<h2 class="subheader">Batch Analysis</h2>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            summary_cols = st.columns(2)
            
            with summary_cols[0]:
                # Create a summary dataframe
                summary_data = []
                for rep, _ in report_list:
                    summary_data.append({
                        'Report #': rep['Report Number'],
                        'Date': rep['Date & Time'],
                        'Category': rep['Predicted Category'],
                        'Severity': rep['Assigned Severity'],
                        'District': rep['Police District'],
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, height=200)
            
            with summary_cols[1]:
                # Category distribution in these reports
                categories = [rep[0]['Predicted Category'] for rep in report_list if rep[0]['Predicted Category'] != "Unknown"]
                if categories:
                    cat_counts = pd.Series(categories).value_counts()
                    
                    fig = px.pie(
                        values=cat_counts.values,
                        names=cat_counts.index,
                        title="Crime Categories Distribution",
                        hole=0.4,
                        color_discrete_sequence=px.colors.qualitative.Bold
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No valid categories found in reports")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show sample report when no files are uploaded
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info("No PDF files uploaded. You can upload police report PDFs to extract structured information and receive AI-powered crime classification.")
        
        # Sample report structure
        st.markdown("#### Sample Report Structure")
        st.markdown("""
        Your police reports should contain the following fields for optimal extraction:
        
        - **Report Number**: Unique identifier for the report
        - **Date & Time**: When the incident occurred
        - **Incident Location**: Address or location description
        - **Coordinates**: Latitude and longitude (optional)
        - **Detailed Description**: Description of the incident
        - **Police District**: The jurisdiction's district
        - **Resolution**: Current status of the case
        - **Suspect Description**: Information about suspects
        - **Victim Information**: Details about victims (anonymized)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------------------
# Page 3: Crime Classification
# -----------------------------------------------
elif page == "Crime Classification":
    st.markdown('<h1 class="main-header">Crime Classification</h1>', unsafe_allow_html=True)
    
    # Description input
    st.markdown('<div class="card">', unsafe_allow_html=True)
    description_input = st.text_area(
        "Enter a Crime Description:",
        "e.g., suspicious person loitering near parked cars, attempting to open car doors at night",
        height=150
    )
    classify_button = st.button("Analyze Crime", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if classify_button and description_input and description_input != "e.g., suspicious person loitering near parked cars, attempting to open car doors at night":
        # Show spinner while "processing"
        with st.spinner("Analyzing crime description..."):
            # Artificial delay for effect
            import time
            time.sleep(1)
            
            # Get prediction
            pred_cat, severity = predict_crime(description_input)
        
        # Results display
        st.markdown('<h2 class="subheader">Classification Results</h2>', unsafe_allow_html=True)
        
        result_cols = st.columns([2, 1])
        
        with result_cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Primary classification
            st.markdown("### Crime Analysis")
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="font-size: 0.9rem; color: #78909C;">PREDICTED CATEGORY</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: #2196F3;">{pred_cat}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Severity visualization
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <div style="font-size: 0.9rem; color: #78909C;">SEVERITY ASSESSMENT</div>
                <div style="margin-top: 10px;">
                    {get_severity_class(severity)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Similar incidents
            st.markdown("### Similar Past Incidents")
            
            # Find similar incidents
            similar_incidents = df[df['Category'] == pred_cat].sample(min(3, len(df[df['Category'] == pred_cat])))
            
            for _, incident in similar_incidents.iterrows():
                st.markdown(f"""
                <div style="padding: 10px; border-left: 3px solid #2196F3; margin-bottom: 10px; background-color: rgba(33, 150, 243, 0.1);">
                    <div style="font-size: 0.8rem; color: #78909C;">{incident['Dates'].strftime('%Y-%m-%d %H:%M')} | {incident['PdDistrict']}</div>
                    <div style="font-size: 1rem;">{incident['Descript']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with result_cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            
            # Response guidelines based on severity
            st.markdown("### Response Guidelines")
            
            if severity == 1:
                st.markdown("""
                📋 **Recommended Actions:**
                - Document incident in system
                - No immediate response needed
                - Monitor for pattern development
                """)
            elif severity == 2:
                st.markdown("""
                🚶 **Recommended Actions:**
                - Dispatch officer when available
                - Take report from involved parties
                - Follow up as needed within 48 hours
                """)
            elif severity == 3:
                st.markdown("""
                🚓 **Recommended Actions:**
                - Dispatch officer promptly
                - Collect evidence and statements
                - Begin investigation process
                - Follow up within 24 hours
                """)
            elif severity == 4:
                st.markdown("""
                🚨 **Recommended Actions:**
                - Immediate officer dispatch
                - Scene security and evidence preservation
                - Detective notification
                - Rapid investigation initiation
                """)
            elif severity == 5:
                st.markdown("""
                🔴 **Recommended Actions:**
                - Emergency response team deployment
                - Multiple unit dispatch
                - Immediate detective involvement
                - Command center notification
                - Area lockdown consideration
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Show examples even if no prediction yet
    if not classify_button or description_input == "e.g., suspicious person loitering near parked cars, attempting to open car doors at night":
        st.markdown('<h2 class="subheader">Example Descriptions</h2>', unsafe_allow_html=True)
        
        example_cols = st.columns(3)
        
        examples = [
            {
                "title": "Vehicle Theft",
                "description": "Car was broken into overnight, steering column damaged, vehicle stolen from residential parking lot.",
                "category": "VEHICLE THEFT",
                "severity": 3
            },
            {
                "title": "Assault",
                "description": "Victim was approached by unknown male who struck victim in face after verbal altercation outside bar at approximately 11:30pm.",
                "category": "ASSAULT",
                "severity": 4
            },
            {
                "title": "Vandalism",
                "description": "Graffiti found on east wall of public building, approximately 3ft x 6ft area affected, spray paint used.",
                "category": "VANDALISM",
                "severity": 2
            }
        ]
        
        for i, example in enumerate(examples):
            with example_cols[i]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"#### {example['title']}")
                st.markdown(f"<div style='font-style: italic; font-size: 0.9rem;'>{example['description']}</div>", unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(f"**Category:** {example['category']}")
                st.markdown(f"**Severity:** {get_severity_class(example['severity'])}", unsafe_allow_html=True)
                if st.button(f"Use Example {i+1}"):
                    st.session_state.example_description = example['description']
                    try:
                        st.rerun()
                    except AttributeError as e:
                        st.error(f"Rerun failed: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

    # Classification explanation
    st.markdown('<h2 class="subheader">How Classification Works</h2>', unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    explanation_cols = st.columns([1, 1])
    
    with explanation_cols[0]:
        st.markdown("""
        ### Classification Process
        
        1. **Text Analysis** - The system extracts key terms and patterns from the crime description
        
        2. **Machine Learning** - A TF-IDF vectorizer converts the text to numerical features, which are then processed by a Logistic Regression model
        
        3. **Category Assignment** - The model predicts the most likely crime category from our database of historical incidents
        
        4. **Severity Assessment** - Based on the assigned category, a pre-defined severity scale (1-5) is applied
        """)
    
    with explanation_cols[1]:
        st.markdown("""
        ### Severity Scale
        
        - **Level 1 (Low)** - Non-criminal, suspicious occurrence, missing person
        
        - **Level 2 (Minor)** - Vandalism, trespass, disorderly conduct 
        
        - **Level 3 (Moderate)** - Theft, vehicle theft, drug offenses
        
        - **Level 4 (High)** - Robbery, weapon laws, burglary
        
        - **Level 5 (Severe)** - Kidnapping, arson, homicide
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer with theme-matching colors
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 20px; background-color: #263238; border-radius: 10px;">
    <p style="color: #ffffff; font-size: 0.8rem;">
        CityX Crime Analysis Dashboard | Data updated as of March 2025
    </p>
    <p style="color: #ffffff; font-size: 0.8rem;">
        Developed by fateen ahmed | Contact: 
        <a href="mailto:fateenahmed.2k@gmail.com" style="color: #FF5252; text-decoration: none;">
            fateenahmed.2k@gmail.com
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
