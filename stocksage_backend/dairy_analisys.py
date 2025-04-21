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


columns_to_drop = [
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
]

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

# Convert date columns to datetime
print("Converting date columns to datetime...")
date_columns = ["Date", "Production Date", "Expiration Date"]
for col in date_columns:
    data[col] = pd.to_datetime(data[col])

# DATE FEATURES

data["Date_Sell"] = np.where(
    data["Date"] > data["Expiration Date"], data["Expiration Date"], data["Date"]
)

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

# Analyze the top 5 rows with the highest values for specific columns
columns_to_analyze = [
    "Quantity_Abble_Sell_Before_Expire",
    "Quantity_Lost",
    "Value_Lost",
    "Sales_Velocity",
]
print(
    data[columns_to_analyze].describe(),
    "\n\n",
)


print(data.columns)
for column in columns_to_analyze:
    print(f"Top 5 rows for {column}:")
    print(
        data.nlargest(5, column)[
            [
                "Days_to_Sell",
                "Product Name",
                "Expire",
                "Quantity (liters/kg)",
                "Quantity Sold (liters/kg)",
                column,
            ]
        ]
    )
    print("\n")

# Count the number of expired and non-expired products
expired_count = data[data["Expire"] == 1].shape[0]
non_expired_count = data[data["Expire"] == 0].shape[0]

print(f"Expired products: {expired_count}")
print(f"Non-expired products: {non_expired_count}")

# Count the number of expired and non-expired products
expired_count = data[
    (data["Price per Unit (sold)"] < data["Price per Unit"]) & (data["Expire"] == 1)
].shape[0]
print(f"Quantidade de produtos com vendido menor que compra: {expired_count}")
