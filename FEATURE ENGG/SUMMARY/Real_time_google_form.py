"""   

1. Set Up Google Form and Response Sheet
Create a Google Form: Design the form to collect the required data.
Enable the Response Sheet: Go to the "Responses" tab in the form and click the green spreadsheet icon. This will create a Google Sheet to store responses.
2. Set Up Google Sheets API
Enable Google Sheets API:
Go to Google Cloud Console.
Create a new project or use an existing one.
Enable the Google Sheets API in the APIs & Services section.
Create Service Account Credentials:
Go to "Credentials" in the Cloud Console.
Click "Create Credentials" → "Service Account".
Assign a role (e.g., Viewer or Editor) and download the JSON key file.
Share the Sheet with the Service Account:
Open the response Google Sheet.
Share it with the service account email from the JSON key file.
3. Install Required Python Libraries
Run the following command to install the required libraries:

 # PYTHON CODE :- 

import gspread
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from oauth2client.service_account import ServiceAccountCredentials
import time

# Authenticate and connect to Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("path_to_your_credentials.json", scope)
client = gspread.authorize(creds)

# Open Google Sheet
sheet = client.open("Your Google Sheet Name").sheet1

# Function to fetch and visualize data
def fetch_and_visualize():
    # Fetch data
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # Visualize data (Customize this section based on your data)
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Your Column Name")  # Replace with your column name
    plt.title("Real-Time Data Visualization")
    plt.xticks(rotation=45)
    plt.show()

# Continuously fetch and update visualization
try:
    while True:
        fetch_and_visualize()
        time.sleep(30)  # Update every 30 seconds
except KeyboardInterrupt:
    print("Visualization stopped.")

    
5. Run the Python Code
Replace "path_to_your_credentials.json" with the path to your JSON credentials file.
Replace "Your Google Sheet Name" with the name of your Google Sheet.
Replace "Your Column Name" in the visualization section with the appropriate column name from your data.
6. Visualize Data
The script will fetch the latest data every 30 seconds and update the visualization.
You can adjust the sleep time to control the update frequency.
Future Enhancements
Dashboards: Use a library like Streamlit or Dash to create an interactive real-time dashboard.
Cloud Deployment: Deploy the script on a cloud platform (e.g., AWS, Google Cloud, or Heroku) for continuous execution.
Advanced Visualizations: Incorporate libraries like Plotly for more interactive and dynamic charts
"""