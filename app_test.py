import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
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
# Streamlit UI: Page Navigation
# -----------------------------------------------
st.title("CityX Crime Analysis Dashboard")

page = st.sidebar.selectbox("Select Page", ["Interactive Map", "Police Report Extraction & Classification", "Crime Classification"])

# -----------------------------------------------
# Page 1: Interactive Map using Folium (Swapped Coordinates)
# -----------------------------------------------
if page == "Interactive Map":
    st.header("Interactive Crime Map")
    # Create a Folium map centered on the average coordinates, with switched order.
    # Now, the center is [mean of Longitude, mean of Latitude]
    map_center = [df['Longitude'].mean(), df['Latitude'].mean()]
    crime_map = folium.Map(location=map_center, zoom_start=12, tiles="CartoDB dark_matter")
    
    # Add a marker cluster.
    marker_cluster = MarkerCluster().add_to(crime_map)
    
    # Optimize: sample up to 1000 markers for performance.
    max_markers = 1000
    if len(df) > max_markers:
        df_map = df.sample(max_markers)
    else:
        df_map = df
        
    # For each record, swap the order: use [Longitude, Latitude] instead of [Latitude, Longitude]
    for idx, row in df_map.iterrows():
        popup_text = f"<strong>Category:</strong> {row['Category']}<br><strong>Description:</strong> {row['Descript']}<br><strong>District:</strong> {row['PdDistrict']}"
        folium.Marker(
            location=[row['Longitude'], row['Latitude']],
            popup=popup_text
        ).add_to(marker_cluster)
    
    # Render the Folium map's HTML directly.
    st.components.v1.html(crime_map._repr_html_(), height=500)

# -----------------------------------------------
# Page 2: Police Report Extraction & Classification
# -----------------------------------------------
elif page == "Police Report Extraction & Classification":
    st.header("Police Report Extraction & Classification")
    st.write("Upload one or more police report PDFs:")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
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
            report_list.append(report)
        
        st.write("Extracted Police Report Data:")
        # Display each report as a vertical (transposed) table.
        # Use a subtle light blue color for highlighting.
        highlight_color = "#ADD8E6"  # light blue
        for i, rep in enumerate(report_list):
            st.subheader(f"Report {i+1}")
            rep_df = pd.DataFrame.from_dict(rep, orient="index", columns=["Value"])
            def highlight_index(val):
                if val in ["Predicted Category", "Assigned Severity"]:
                    return f"background-color: {highlight_color}"
                else:
                    return ""
            rep_df_styled = rep_df.style.apply(lambda s: [highlight_index(s.name)] * len(s), axis=1)
            # Optionally, also display a snippet of the PDF text.
            st.table(rep_df_styled)
            # Show a snippet (first 300 characters) of the extracted PDF text.
            st.text_area("PDF Text Snippet", text[:300] + "..." if len(text) > 300 else text, height=150)
    else:
        st.write("No PDF files uploaded.")

# -----------------------------------------------
# Page 3: Crime Classification
# -----------------------------------------------
elif page == "Crime Classification":
    st.header("Crime Classification")
    description_input = st.text_area("Enter a Crime Description:", "e.g., suspicious person loitering near parked cars")
    if st.button("Predict Crime Category & Severity"):
        pred_cat, severity = predict_crime(description_input)
        st.write("**Predicted Crime Category:**", pred_cat)
        st.write("**Assigned Severity Level:**", severity)
