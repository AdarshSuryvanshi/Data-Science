"""1. Introduction to Pandas
Pandas is a Python library designed for data manipulation and analysis. It provides two main data structures:

Series: One-dimensional arrays.
DataFrame: Two-dimensional, tabular data.
Real-World Analogy: Think of a Series as a single column in an Excel sheet, and a DataFrame as the entire Excel table.

Example:
python
Copy code
import pandas as pd

# Creating a Series
sales = pd.Series([500, 700, 900], index=['January', 'February', 'March'])
print(sales)

# Creating a DataFrame
data = {
    'Product': ['A', 'B', 'C'],
    'Sales': [500, 700, 900],
    'Profit': [50, 70, 90]
}
df = pd.DataFrame(data)
print(df)
2. Creating DataFrames
From Dictionary:
python
Copy code
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)
print(df)
From Lists:
python
Copy code
data = [
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Los Angeles'],
    ['Charlie', 35, 'Chicago']
]
df = pd.DataFrame(data, columns=['Name', 'Age', 'City'])
print(df)
Real-World Use:
Loading data from a CSV file (common in analytics):

python
Copy code
df = pd.read_csv('sales_data.csv')
print(df.head())  # Show the first 5 rows
3. Data Wrangling
Data Wrangling involves cleaning and transforming data into a usable format.

3.1 Sorting Data
python
Copy code
# Sort rows by a column
df_sorted = df.sort_values('Age')
print(df_sorted)

# Sort in descending order
df_sorted_desc = df.sort_values('Age', ascending=False)
print(df_sorted_desc)
3.2 Renaming Columns
python
Copy code
df.rename(columns={'Age': 'Years', 'City': 'Location'}, inplace=True)
print(df)
3.3 Resetting and Sorting Index
python
Copy code
# Reset the index
df_reset = df.reset_index(drop=True)
print(df_reset)

# Sort index
df_sorted_index = df.sort_index()
print(df_sorted_index)
4. Subsetting Data
By Rows:
python
Copy code
# Select rows where Age > 30
subset = df[df['Age'] > 30]
print(subset)
By Columns:
python
Copy code
# Select specific columns
subset_columns = df[['Name', 'City']]
print(subset_columns)
By Rows and Columns:
python
Copy code
# Select specific rows and columns
subset = df.loc[df['Age'] > 30, ['Name', 'City']]
print(subset)
5. Handling Missing Data
Real-world datasets often have missing values. Pandas provides tools to handle them.

Check for Missing Data:
python
Copy code
print(df.isnull())  # Check for missing values
print(df.isnull().sum())  # Count missing values per column
Drop Missing Data:
python
Copy code
df_cleaned = df.dropna()  # Drop rows with missing values
print(df_cleaned)
Fill Missing Data:
python
Copy code
df_filled = df.fillna('Unknown')  # Replace missing values
print(df_filled)
6. Combining DataFrames
Concatenation:
python
Copy code
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
df_combined = pd.concat([df1, df2])
print(df_combined)
Merging:
python
Copy code
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Age': [25, 30]})
df_merged = pd.merge(df1, df2, on='ID')
print(df_merged)
7. Summarizing Data
Basic Statistics:
python
Copy code
print(df.describe())  # Summary statistics
print(df['Age'].mean())  # Mean of a column
Grouping:
python
Copy code
grouped = df.groupby('City')['Age'].mean()
print(grouped)
8. Advanced Operations
Reshaping Data:
python
Copy code
# Pivot (reshape rows into columns)
df_pivot = df.pivot(index='Name', columns='City', values='Age')
print(df_pivot)
Melting Data:
python
Copy code
# Melt (reshape columns into rows)
df_melt = pd.melt(df, id_vars='Name', value_vars=['Age', 'City'])
print(df_melt)
9. Real-World Example: Sales Analysis
Dataset:
python
Copy code
data = {
    'Month': ['January', 'February', 'March'],
    'Product': ['A', 'B', 'C'],
    'Sales': [300, 400, 500],
    'Profit': [30, 50, 70]
}
df = pd.DataFrame(data)"""


"""  
Plan for Learning Pandas and EDA:
Introduction to Pandas:

What is Pandas?
Why is it used in data analysis and EDA?
Tidy Data Principles:

What is tidy data?
Why is it important for analysis?
DataFrames:

Creating DataFrames from scratch, dictionaries, lists, or files.
Adding, renaming, and removing columns and rows.
Reshaping Data:

Melting, pivoting, and transposing data.
Sorting and reindexing rows and columns.
Selecting and Filtering Data:

Using loc, iloc, at, and iat for data selection.
Logical filtering and querying data.
Handling Missing Data:

Detecting missing values (isnull, notnull).
Dropping or filling missing data.
Combining Datasets:

Merging, concatenating, and joining datasets.
Types of joins and their use cases.
Summarizing Data:

Descriptive statistics (mean, sum, count, etc.).
Grouping and aggregating data.
Vectorized Operations:

Performing arithmetic and logical operations on DataFrames.
Advanced Concepts:

Working with MultiIndex.
Using regular expressions for filtering.
Binning data with pd.qcut.
Real-World Examples:

Analyzing a dataset (e.g., Titanic, sales data, etc.).
Step-by-step walkthrough of EDA.
Cheat Sheet Reference:

Using a Pandas cheat sheet effectively.

"""
"""

1. Introduction to Pandas
What is Pandas?
Key Features
Installation and Setup

2. Pandas Basics
DataFrames and Series
Creating DataFrames (pd.DataFrame)
Creating Series (pd.Series)
Loading and Saving Data
CSV, Excel, JSON, SQL, etc.

3. Data Wrangling
Tidy Data Concepts
Rows as observations
Columns as variables
Manipulating Rows and Columns
Adding, renaming, and removing
Reindexing

Sorting Data

4. Subsetting and Filtering
Selecting Data by Position and Labels
.iloc[], .loc[], .at[], .iat[]
Using Boolean Indexing
Logical conditions for filtering rows

5. Reshaping and Pivoting
Wide and Long Data
pd.melt()
pd.pivot(), pd.pivot_table()
Concatenating and Merging
pd.concat()
pd.merge()
Joins (inner, outer, left, right)

6. Handling Missing Data
Identifying Missing Values
pd.isnull(), pd.notnull()
Dealing with Missing Data
Dropping rows/columns (dropna())
Filling missing values (fillna())

7. Summarizing and Aggregating
Descriptive Statistics
sum(), mean(), min(), max(), count(), etc.
Grouping Data
groupby()
Aggregations and transformations

8. Advanced Operations
Method Chaining
Vectorized Operations
Applying Functions
apply(), map(), applymap()

9. Real-World Examples
Working with Sales Data
Analyzing Weather Data
Exploring E-Commerce Data

10. Cheat Sheets and Guides
Use simplified cheat sheets for quick access to Pandas syntax and operations.
First Step: Pandas Introduction
1. What is Pandas?
Pandas is a Python library used for data manipulation and analysis. It provides two main data structures:

DataFrame: A 2D table-like data structure with labeled rows and columns.
Series: A one-dimensional array-like structure with labeled indices.
Why Use Pandas?
Easy to clean and prepare data.
Powerful for exploration and transformation.
Supports reading and writing in various formats (CSV, Excel, SQL, etc.).
Integrates seamlessly with other libraries like NumPy, Matplotlib, and Seaborn.

pip install pandas
2. Creating DataFrames
A DataFrame is like a table in a database or a spreadsheet in Excel. Here's how to create one:


import pandas as pd

# Creating a simple DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "San Francisco", "Los Angeles"]
}

df = pd.DataFrame(data)
print(df)
Output:


      Name  Age           City
0    Alice   25      New York
1      Bob   30  San Francisco
2  Charlie   35   Los Angeles
3. Loading and Saving Data
Loading a CSV File:

python
Copy
Edit
df = pd.read_csv("data.csv")
print(df.head())  # View the first 5 rows
Saving a DataFrame to CSV:

python
Copy
Edit
df.to_csv("output.csv", index=False)




. What is Pandas?
Pandas is a Python library used for data analysis and manipulation. It provides two main data structures:

Series: A one-dimensional array with labels (like a column in a table).
DataFrame: A two-dimensional, tabular data structure (like an Excel spreadsheet).
Think of Pandas as your assistant for cleaning, organizing, and analyzing data, whether it's for exploring trends or preparing it for machine learning models.

Installing Pandas
bash
Copy
Edit
pip install pandas
2. Creating a DataFrame
A DataFrame is like an Excel sheet. Let’s create one:

Example: Manual Creation
python
Copy
Edit
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "City": ["New York", "Los Angeles", "Chicago"]
}

df = pd.DataFrame(data)
print(df)
Output:

markdown
Copy
Edit
      Name  Age         City
0    Alice   25     New York
1      Bob   30  Los Angeles
2  Charlie   35      Chicago
3. Loading Data
Pandas makes it easy to load data from various formats like CSV, Excel, SQL, etc.

Example: Loading a CSV File
python
Copy
Edit
df = pd.read_csv("your_file.csv")
print(df.head())  # Shows the first 5 rows
4. Inspecting Data
Use the following methods to understand your dataset:

.head(): View the first 5 rows.
.tail(): View the last 5 rows.
.shape: Get the number of rows and columns.
.info(): Summary of the DataFrame.
.describe(): Statistical summary of numerical columns.
Example:
python
Copy
Edit
print(df.info())
print(df.describe())
5. Tidy Data
Tidy Data means:

Each column is a variable.
Each row is an observation.
Reshaping Data:
Pandas makes reshaping easy with methods like:

pd.melt(): Converts columns into rows.
pd.pivot(): Converts rows into columns.
Example: Melting Data (Wide to Long)
python
Copy
Edit
data = {
    "ID": [1, 2],
    "Math": [85, 90],
    "Science": [78, 88]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Melting
df_melted = pd.melt(df, id_vars=["ID"], var_name="Subject", value_name="Score")
print("\nMelted DataFrame:")
print(df_melted)
Output:

javascript
Copy
Edit
Original DataFrame:
   ID  Math  Science
0   1    85       78
1   2    90       88

Melted DataFrame:
   ID  Subject  Score
0   1     Math     85
1   2     Math     90
2   1  Science     78
3   2  Science     88
Example: Pivoting Data (Long to Wide)

# Pivoting back
df_pivoted = df_melted.pivot(index="ID", columns="Subject", values="Score")
print("\nPivoted DataFrame:")
print(df_pivoted)
Output:

javascript

Subject  Math  Science
ID                    
1         85       78
2         90       88
6. Sorting Data
df.sort_values(by='column_name'): Sort rows by a column.
df.sort_index(): Sort rows by index.
Example:

df_sorted = df.sort_values(by="Age", ascending=False)
print(df_sorted)
7. Filtering Data
Logical Filtering: Use conditions to filter rows.
Example:

# People older than 28
filtered_df = df[df["Age"] > 28]
print(filtered_df)
8. Handling Missing Data
.dropna(): Remove rows with missing values.
.fillna(value): Fill missing values.
Example:

data = {
    "Name": ["Alice", "Bob", "Charlie", None],
    "Age": [25, None, 35, 40]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Fill missing values
df_filled = df.fillna({"Name": "Unknown", "Age": df["Age"].mean()})
print("\nFilled DataFrame:")
print(df_filled)
9. Combining DataFrames
Use pd.concat() or pd.merge() to combine multiple DataFrames.

Example: Concatenation

df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"A": [5, 6], "B": [7, 8]})

combined_df = pd.concat([df1, df2])
print(combined_df)
10. GroupBy and Aggregations
Group data based on a column and apply aggregate functions.

Example:

data = {
    "Category": ["A", "B", "A", "B"],
    "Values": [10, 20, 30, 40]
}

df = pd.DataFrame(data)
grouped = df.groupby("Category").sum()
print(grouped)
Output:


          Values
Category        
A             40
B             60


1. Understand the Dataset
Start by loading and inspecting the dataset to get an overview of its structure and contents.

Example: Loading a Dataset
We'll use a dataset called sales.csv as an example.

python
Copy
Edit
import pandas as pd

# Load the dataset
df = pd.read_csv("sales.csv")

# Check the first few rows
print(df.head())

# Get a summary
print(df.info())
2. Summary Statistics
Use .describe() to generate statistical summaries of numerical columns.


# Summary statistics
print(df.describe())
This provides details like mean, median, min, max, and standard deviation.

3. Check for Missing Values
Missing values can skew your analysis. Identify them using .isnull() and .sum().


# Check for missing values
print(df.isnull().sum())

# Fill missing values (example)
df_filled = df.fillna({"column_name": "default_value"})
4. Data Visualization
Visualizations provide insights that are hard to extract from raw data. Use libraries like Matplotlib and Seaborn.

Example: Import Libraries


import matplotlib.pyplot as plt
import seaborn as sns
a. Histograms (Distribution of a Single Variable)

# Plot a histogram
df['column_name'].hist(bins=10)
plt.title("Histogram of Column")
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()
b. Boxplots (Detecting Outliers)


# Boxplot for a column
sns.boxplot(data=df, x="column_name")
plt.title("Boxplot of Column")
plt.show()
c. Correlation Matrix
Correlation shows relationships between numerical columns.

# Correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()
d. Scatter Plots (Relationship Between Two Variables)


# Scatter plot
sns.scatterplot(data=df, x="column1", y="column2")
plt.title("Scatter Plot")
plt.show()
5. Handle Outliers
Outliers can distort analysis. Use boxplots or scatterplots to detect them.

Example: Removing Outliers

# Remove outliers based on a condition
df_filtered = df[df['column_name'] < threshold]
6. Categorical Data Analysis
For columns with categories, analyze value counts and proportions.


# Value counts
print(df['category_column'].value_counts())

# Visualize with a bar chart
df['category_column'].value_counts().plot(kind='bar')
plt.title("Category Distribution")
plt.show()
7. Feature Engineering
Create new features to enhance your dataset.

Example: Adding a New Column

# Add a new column based on existing data
df['Total_Sales'] = df['Unit_Price'] * df['Quantity']
8. Data Transformation
Transform data for better analysis or model performance.

Example: Normalization

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df['normalized_column'] = scaler.fit_transform(df[['column_name']])
9. Export Cleaned Dataset
Save the cleaned and prepared dataset for further use.

# Save to a new CSV
df.to_csv("cleaned_sales.csv", index=False)




Time-Series Analysis for Trends and Forecasting
1. What is Time-Series Analysis?
Time-series data refers to observations collected at regular time intervals (e.g., daily stock prices, monthly sales, yearly disease cases). The goal is to:

Analyze trends, seasonality, and cyclic patterns.
Forecast future values using past data.
2. Essential Components of Time-Series Data
Trend: Overall upward or downward movement.
Seasonality: Regular patterns repeating at specific intervals (e.g., monthly or yearly).
Cyclic Patterns: Long-term oscillations not tied to fixed intervals.
Noise: Random variations not explained by other components.
3. Tools and Libraries for Time-Series Analysis
Python Libraries: pandas, matplotlib, seaborn, statsmodels, fbprophet, scikit-learn.
Install required libraries:

bash
Copy
Edit
pip install matplotlib seaborn statsmodels prophet
4. Steps for Time-Series Analysis
Step 1: Load the Dataset
python
Copy
Edit
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('time_series_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Display the first few rows
print(df.head())
Step 2: Visualize the Time-Series Data
python
Copy
Edit
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Value'], label="Observed Data")
plt.title("Time-Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()
Step 3: Decompose the Time-Series
Use seasonal decomposition to identify trends, seasonality, and residuals.

python
Copy
Edit
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['Value'], model='additive')
decomposition.plot()
plt.show()
Step 4: Check for Stationarity
Stationarity means that statistical properties (mean, variance) of the time series remain constant over time.

Augmented Dickey-Fuller (ADF) Test:
python
Copy
Edit
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['Value'])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")
if result[1] < 0.05:
    print("The series is stationary.")
else:
    print("The series is not stationary.")
Step 5: Make the Series Stationary
Apply transformations like differencing, log transformations, or smoothing if the series is not stationary.

python
Copy
Edit
df['Differenced'] = df['Value'] - df['Value'].shift(1)
df['Differenced'].dropna().plot(title="Differenced Data")
plt.show()
Step 6: Build a Forecasting Model
ARIMA (AutoRegressive Integrated Moving Average):
python
Copy
Edit
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(df['Value'], order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=10)
print(forecast)
Prophet (by Facebook):
python
Copy
Edit
from prophet import Prophet

# Prepare data for Prophet
prophet_df = df.reset_index()[['Date', 'Value']]
prophet_df.columns = ['ds', 'y']

# Fit model
model = Prophet()
model.fit(prophet_df)

# Forecast
future = model.make_future_dataframe(periods=10)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.show()
5. Example Use Cases
Healthcare: Forecast monthly hospital admissions based on past trends.
Sales: Predict future revenue for better inventory management.
Finance: Estimate stock prices for portfolio optimization.
Energy: Predict electricity demand for load balancing.
6. Next Steps
Would you like to:

Work through a sample dataset for practice?
Explore specific forecasting models, like ARIMA or Prophet, in more detail?
Dive into domain-specific case studies, such as sales or healthcare predictions?
Let me know your preference!














ChatGPT can make mistakes. Check important info.



"""

