import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

"""
THIS SCRIPT IS FOR DATA PREPROCESSING OF A DAIRY PRODUCTS DATASET
the 'Production Date' is the date that it was bought from the farm,
the 'Expiration Date' is the last date that it should be sold,
so sometimes de 'Date' exceeds the 'Expiration Date', that means we lost products 
(Quantity in Stock), and considere that 'Date_Sell', being the 'Expiration Date', 
but if the 'Date' is lower than the expiration date, we still have time to sell it.
"""


def remove_outliers(df, column, threshold=1.5):
    """
    Remove outliers de uma coluna numérica usando o método IQR.

    Parâmetros:
    - df: DataFrame do pandas.
    - column: Nome da coluna para remover outliers.
    - threshold: Multiplicador do IQR (padrão: 1.5, pode ser ajustado).

    Retorna:
    - DataFrame sem os outliers da coluna especificada.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR

    # Filtra os valores dentro dos limites
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return filtered_df


# Create directories for outputs
os.makedirs("dairy/dairy_plots", exist_ok=True)
os.makedirs("dairy/dairy_models", exist_ok=True)
os.makedirs("dairy/dairy_data", exist_ok=True)

# Load the dataset
print("Loading dataset...")
dados_path = r"C:\Users\alexa\OneDrive\Anexos\Fiap\projeto_fase4\Enterprise Challenge\data_analysis\stock_control\dairy_dataset.csv"
data = pd.read_csv(dados_path)

# Display basic information
print(f"Dataset shape: {data.shape}")


data["Date_Sell"] = np.where(
    data["Date"] > data["Expiration Date"], data["Expiration Date"], data["Date"]
)

# Convert date columns to datetime
print("Converting date columns to datetime...")
date_columns = ["Date", "Production Date", "Expiration Date", "Date_Sell"]
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

# Extract time-based features
print("Extracting time-based features...")
data["Year"] = data["Date_Sell"].dt.year
data["Month"] = data["Date_Sell"].dt.month
data["Day"] = data["Date_Sell"].dt.day
data["DayOfWeek"] = data["Date_Sell"].dt.dayofweek
data["Quarter"] = data["Date_Sell"].dt.quarter
# data["Season"] = (data["Date_Sell"] % 12 + 3) // 3

# Calculate additional features
print("Calculating additional features...")

# Days to sell is to know how many days we got to sell the product
data["Days_to_Sell"] = (data["Date_Sell"] - data["Production Date"]).dt.days
data["Days_to_Expire"] = (data["Expiration Date"] - data["Date_Sell"]).dt.days

# boleans to know if the product is expired or not (0 = false = not expired, 1 = true = expired)
condition = (data["Expiration Date"] == data["Date_Sell"]) & (
    data["Quantity in Stock (liters/kg)"] != 0
)
data["Expire"] = np.where(condition, 1, 0)

# QUANTITY FEATURES

data["Sales_Velocity"] = np.where(
    data["Days_to_Sell"] > 0,
    data["Quantity Sold (liters/kg)"] / data["Days_to_Sell"],
    data["Quantity Sold (liters/kg)"],  # Caso Days_to_Sell seja 0
)
data = remove_outliers(data, "Sales_Velocity")
print(f"Dataset shape after removing Outliers: {data.shape}")

# Calculate the quantity that still can be sold with the remaning days to expire
data["Quantity_Abble_Sell_Before_Expire"] = (
    data["Sales_Velocity"] * data["Days_to_Expire"]
)

# Calculate the quantity that was lost by expiration date
data["Quantity_Lost"] = np.where(
    data["Quantity_Abble_Sell_Before_Expire"] >= data["Quantity in Stock (liters/kg)"],
    0,
    data["Quantity in Stock (liters/kg)"] - data["Quantity_Abble_Sell_Before_Expire"],
)

data["Stock_Efficiency"] = (data["Sales_Velocity"] * data["Shelf Life (days)"]) / data[
    "Quantity (liters/kg)"
]

# PRICE FEATURES
# Count the number of rows where "Price per Unit (sold)" is less than "Price per Unit"
price_comparison_count = data[
    data["Price per Unit (sold)"] < data["Price per Unit"]
].shape[0]
print(
    f"Number of rows where 'Price per Unit (sold)' is less than 'Price per Unit': {price_comparison_count}"
)

# Because the most of the price per unit sold is lower than the price per unit,
# we are going to use a patter of the price per unit sold, calculated based on the price per unit
data["Price per Unit (sold)"] = data["Price per Unit"] * 1.25

data["Value_Lost"] = data["Quantity_Lost"] * data["Price per Unit"]

data["Revenue_Before_Losses"] = (
    data["Price per Unit (sold)"] - data["Price per Unit"]
) * data["Quantity Sold (liters/kg)"]

data["Real_Revenue"] = data["Revenue_Before_Losses"] - data["Value_Lost"]


# Visualize key relationships
print("Creating visualizations...")

# Plot 1: Quantity Sold vs Days to Sell
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x="Days_to_Sell", y="Quantity Sold (liters/kg)", hue="Product Name", data=data
)
plt.title("Quantity Sold vs Days to Sell by Product")
plt.tight_layout()
plt.savefig("dairy/dairy_plots/quantity_sold_vs_days_to_sell.png")
plt.close()

# Plot 2: Sales by Product
plt.figure(figsize=(12, 6))
product_sales = (
    data.groupby("Product Name")["Quantity Sold (liters/kg)"]
    .sum()
    .sort_values(ascending=False)
)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Total Sales by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_plots/sales_by_product.png")
plt.close()

# Plot 2.1: Revenue by Product
plt.figure(figsize=(12, 6))
product_sales = (
    data.groupby("Product Name")["Real_Revenue"].sum().sort_values(ascending=False)
)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Total Revenue by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_plots/revenue_by_product.png")
plt.close()

# Plot 2.2: Velocity by Product
plt.figure(figsize=(12, 6))
product_sales = (
    data.groupby("Product Name")["Sales_Velocity"].sum().sort_values(ascending=False)
)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Sales_Velocity by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_plots/sales_velocity_by_product.png")
plt.close()

# Plot 2.3: Quantity Lost by Product
plt.figure(figsize=(12, 6))
product_sales = (
    data.groupby("Product Name")["Quantity_Lost"].sum().sort_values(ascending=False)
)
sns.barplot(x=product_sales.index, y=product_sales.values)
plt.title("Quantity_Lost by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_plots/quantity_lost_by_product.png")
plt.close()

# Plot 3: Sales by Month
plt.figure(figsize=(10, 6))
monthly_sales = data.groupby("Month")["Quantity Sold (liters/kg)"].sum()
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("dairy/dairy_plots/monthly_sales.png")
plt.close()

# Plot 3.1: Revenue by Month
plt.figure(figsize=(10, 6))
monthly_sales = data.groupby("Month")["Real_Revenue"].sum()
sns.lineplot(x=monthly_sales.index, y=monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Quantity Sold")
plt.xticks(range(1, 13))
plt.tight_layout()
plt.savefig("dairy/dairy_plots/monthly_revenue.png")
plt.close()

# Plot 4: Stock Efficiency by Product
plt.figure(figsize=(12, 6))
stock_efficiency = (
    data.groupby("Product Name")["Stock_Efficiency"].mean().sort_values(ascending=False)
)
sns.barplot(x=stock_efficiency.index, y=stock_efficiency.values)
plt.title("Average Stock Efficiency by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_plots/stock_efficiency_by_product.png")
plt.close()

# Plot 5: Correlation Matrix of Numerical Features
plt.figure(figsize=(16, 12))
numerical_cols = data.select_dtypes(include=["float64", "int64"]).columns
correlation = data[numerical_cols].corr()
mask = np.triu(np.ones_like(correlation, dtype=bool))
sns.heatmap(correlation, mask=mask, annot=False, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.tight_layout()
plt.savefig("dairy/dairy_plots/correlation_matrix.png")
plt.close()

# Prepare features for modeling
print("Preparing features for modeling...")

# Define target variable and features
target = "Real_Revenue"

# Features to drop
drop_cols = [
    "Location",
    "Total Land Area (acres)",
    "Number of Cows",
    "Farm Size",
    "Brand",
    "Storage Condition",
    "Approx. Total Revenue(INR)",
    "Customer Location",
    "Sales Channel",
    "Minimum Stock Threshold (liters/kg)",
    "Reorder Quantity (liters/kg)",
    "Product ID",
    target,
]

# Categorical features to encode
categorical_features = [
    # "Location",
    # "Farm Size",
    "Product Name",
    # "Brand",
    # "Storage Condition",
    # "Customer Location",
    # "Sales Channel",
]

# Numerical features to scale
numerical_features = [
    col
    for col in data.columns
    if col not in categorical_features + drop_cols + date_columns
    and data[col].dtype in ["int64", "float64"]
]

# Create feature sets
X = data.drop(drop_cols + date_columns, axis=1)
y = data[target]

# Print feature information
print(f"Target variable: {target}")
print(f"Number of categorical features: {len(categorical_features)}")
print(f"Number of numerical features: {len(numerical_features)}")

# Save the list of features for later use
feature_lists = {
    "categorical_features": categorical_features,
    "numerical_features": numerical_features,
    "target": target,
}

with open("dairy/dairy_data/feature_lists.pkl", "wb") as f:
    pickle.dump(feature_lists, f)

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Save the original test data for later evaluation
test_data = data.loc[X_test.index].copy()
test_data.to_csv("dairy/dairy_data/test_data.csv", index=False)

# Fit the preprocessor on the training data
print("Fitting preprocessor...")
preprocessor.fit(X_train)

# Save the preprocessor
with open("dairy/dairy_models/preprocessor.pkl", "wb") as f:
    pickle.dump(preprocessor, f)

# Transform the data
print("Transforming data...")
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save the processed data
print("Saving processed data...")
np.save("dairy/dairy_data/X_train_processed.npy", X_train_processed)
np.save("dairy/dairy_data/X_test_processed.npy", X_test_processed)
np.save("dairy/dairy_data/y_train.npy", y_train.values)
np.save("dairy/dairy_data/y_test.npy", y_test.values)

# Save the original feature data for reference
X_train.to_csv("dairy/dairy_data/X_train_original.csv", index=False)
X_test.to_csv("dairy/dairy_data/X_test_original.csv", index=False)

print("Data preprocessing completed successfully!")
