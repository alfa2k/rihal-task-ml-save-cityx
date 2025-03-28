# 2025 ML Rihal Codestacker Challenge

## Project Components

### Level 1: Exploratory Data Analysis (EDA)
- **Data Cleaning & Preprocessing:**  
  Clean and standardize the raw crime dataset.
- **Trend Analysis:**  
  Extract key insights and visualize crime trends over time.

### Level 2: Crime Classification & Severity Assignment
- **Crime Type Prediction:**  
  Build a text classifier using crime descriptions to predict the crime category.
- **Severity Assignment:**  
  Map each predicted category to a severity level (1-5) based on predefined rules.

### Level 3: Geo-Spatial Mapping & Basic Web UI
- **Interactive Mapping:**  
  Visualize crime incidents on interactive maps using marker clustering and heatmaps.
- **Dashboard Interface:**  
  Create a user-friendly web interface using Streamlit for exploring the data.

### Level 4: Advanced Web UI & Report Extraction
- **PDF Report Extraction:**  
  Automatically extract key fields from police report PDFs.
- **Integration:**  
  Feed the extracted data into the classifier to predict crime category and severity.

### Bonus Task: Deployment
- **Cloud Deployment:**  
  Deployed the application on a free cloud platform - Streamlit Community Cloud so that the web UI and inference pipeline are accessible via a public URL.

## Project Structure

```
├── app.py                   # Streamlit dashboard code
├── analysis.ipynb           # Jupyter Notebook with EDA and modeling
├── Competition_Dataset.csv  # Crime dataset
├── requirements.txt         # Python dependencies
```

## Running Locally

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the App:**

   ```bash
   streamlit run app.py
   ```

   The dashboard will be available at [http://localhost:8501](http://localhost:8501).

## Deployment

The app is deployed publicly at:  
[CityX Crime Analysis Dashboard](https://rihal-task-ml-save-cityx-mnyynnfsfsrbw8dtdyip8n.streamlit.app/)

[Wake the streamlit app in case it's gone to sleep 😅]

## Contact

Developed by **fateen ahmed**  
Email: [fateenahmed.2k@gmail.com](mailto:fateenahmed.2k@gmail.com)
