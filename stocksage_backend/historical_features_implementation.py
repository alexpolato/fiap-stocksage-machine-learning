import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script implements historical feature engineering for dairy stock management.
It creates various time-based features to improve prediction of:
1. Expiration risk - probability that a product will expire before being sold
2. Optimal reorder quantity - how much to order based on sales trends

The historical features include:
- Rolling window sales velocities (7, 14, 30, 90 days)
- Exponential weighted moving averages
- Lag features for seasonality detection
- Product-specific metrics
"""


def load_data(file_path):
    """
    Load the dairy dataset and perform basic preprocessing
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Convert date columns to datetime
    date_columns = ["Date", "Production Date", "Expiration Date"]
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])

    # Sort by product and date for time series analysis
    data = data.sort_values(["Product Name", "Date"])

    return data


def create_base_features(data):
    """
    Create base features needed for historical feature engineering
    """
    print("Creating base features...")

    # Date when product is sold or expires (whichever comes first)
    data["Date_Sell"] = np.where(
        data["Date"] > data["Expiration Date"], data["Expiration Date"], data["Date"]
    )

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_dtype(data["Date_Sell"]):
        data["Date_Sell"] = pd.to_datetime(data["Date_Sell"])

    # Days to sell and expire
    data["Days_to_Sell"] = (data["Date_Sell"] - data["Production Date"]).dt.days
    data["Days_to_Expire"] = (data["Expiration Date"] - data["Date_Sell"]).dt.days

    # Calculate sales velocity (units sold per day)
    data["Sales_Velocity"] = np.where(
        data["Days_to_Sell"] > 0,
        data["Quantity Sold (liters/kg)"] / data["Days_to_Sell"],
        data["Quantity Sold (liters/kg)"],  # If Days_to_Sell is 0
    )

    # Expiration flag (1 if product expired before being completely sold, 0 otherwise)
    data["Expired"] = (
        (data["Expiration Date"] <= data["Date"])
        & (data["Quantity in Stock (liters/kg)"] > 0)
    ).astype(int)

    # Extract time-based features
    data["Year"] = data["Date_Sell"].dt.year
    data["Month"] = data["Date_Sell"].dt.month
    data["Day"] = data["Date_Sell"].dt.day
    data["DayOfWeek"] = data["Date_Sell"].dt.dayofweek
    data["Quarter"] = data["Date_Sell"].dt.quarter
    data["WeekOfYear"] = data["Date_Sell"].dt.isocalendar().week

    return data


def create_product_historical_features(data):
    """
    Create historical features for each product
    """
    print("Creating product historical features...")

    # Create a copy to avoid modifying the original
    result_df = data.copy()

    # List to store all product dataframes
    all_product_dfs = []

    # Process each product separately
    for product in data["Product Name"].unique():
        print(f"  Processing product: {product}")

        # Filter data for this product
        product_data = data[data["Product Name"] == product].copy()

        # Sort by date
        product_data = product_data.sort_values("Date_Sell")

        # Create daily resampled data for this product
        daily_data = create_daily_product_data(product_data)

        # Create rolling window features
        daily_data = create_rolling_window_features(daily_data)

        # Create exponential weighted features
        daily_data = create_ewm_features(daily_data)

        # Create lag features
        daily_data = create_lag_features(daily_data)

        # Create seasonality features
        daily_data = create_seasonality_features(daily_data)

        # Reset index to get Date as a column
        daily_data = daily_data.reset_index()

        # Add product name
        daily_data["Product Name"] = product

        # Store this product's processed data
        all_product_dfs.append(daily_data)

    # Combine all product dataframes
    historical_features_df = pd.concat(all_product_dfs, ignore_index=True)

    # Merge historical features back to original data
    result_df = pd.merge(
        result_df, historical_features_df, on=["Product Name", "Date_Sell"], how="left"
    )

    return result_df


def create_daily_product_data(product_data):
    """
    Create daily resampled data for a product
    """
    # Set date as index
    product_data = product_data.set_index("Date_Sell")

    # Create daily series for key metrics
    daily_sales = (
        product_data["Quantity Sold (liters/kg)"].resample("D").sum().fillna(0)
    )
    daily_stock = (
        product_data["Quantity in Stock (liters/kg)"]
        .resample("D")
        .mean()
        .fillna(method="ffill")
    )
    daily_price = (
        product_data["Price per Unit"].resample("D").mean().fillna(method="ffill")
    )
    daily_price_sold = (
        product_data["Price per Unit (sold)"]
        .resample("D")
        .mean()
        .fillna(method="ffill")
    )

    # Combine into a dataframe
    daily_data = pd.DataFrame(
        {
            "Daily_Sales": daily_sales,
            "Daily_Stock": daily_stock,
            "Daily_Price": daily_price,
            "Daily_Price_Sold": daily_price_sold,
        }
    )

    # Calculate daily sales velocity
    daily_data["Daily_Sales_Velocity"] = daily_data["Daily_Sales"]

    return daily_data


def create_rolling_window_features(daily_data):
    """
    Create rolling window features for sales velocity
    """
    # Define window sizes
    windows = [7, 14, 30, 90]

    for window in windows:
        # Sales rolling sum
        daily_data[f"Sales_Sum_{window}d"] = (
            daily_data["Daily_Sales"].rolling(window=window, min_periods=1).sum()
        )

        # Sales velocity (average daily sales)
        daily_data[f"Sales_Velocity_{window}d"] = (
            daily_data[f"Sales_Sum_{window}d"] / window
        )

        # Stock level rolling average
        daily_data[f"Stock_Avg_{window}d"] = (
            daily_data["Daily_Stock"].rolling(window=window, min_periods=1).mean()
        )

        # Price rolling average
        daily_data[f"Price_Avg_{window}d"] = (
            daily_data["Daily_Price"].rolling(window=window, min_periods=1).mean()
        )

        # Price sold rolling average
        daily_data[f"Price_Sold_Avg_{window}d"] = (
            daily_data["Daily_Price_Sold"].rolling(window=window, min_periods=1).mean()
        )

        # Sales volatility (standard deviation)
        daily_data[f"Sales_Volatility_{window}d"] = (
            daily_data["Daily_Sales"].rolling(window=window, min_periods=3).std()
        )

    return daily_data


def create_ewm_features(daily_data):
    """
    Create exponential weighted moving average features
    """
    # Define span values
    spans = [7, 14, 30]

    for span in spans:
        # EWM for sales
        daily_data[f"Sales_EWM_{span}d"] = (
            daily_data["Daily_Sales"].ewm(span=span, adjust=False).mean()
        )

        # EWM for sales velocity
        daily_data[f"Sales_Velocity_EWM_{span}d"] = (
            daily_data["Daily_Sales_Velocity"].ewm(span=span, adjust=False).mean()
        )

        # EWM for stock
        daily_data[f"Stock_EWM_{span}d"] = (
            daily_data["Daily_Stock"].ewm(span=span, adjust=False).mean()
        )

    return daily_data


def create_lag_features(daily_data):
    """
    Create lag features for sales and stock
    """
    # Define lag periods
    lags = [1, 2, 3, 7, 14, 30]

    for lag in lags:
        # Lag for sales
        daily_data[f"Sales_Lag_{lag}d"] = daily_data["Daily_Sales"].shift(lag)

        # Lag for stock
        daily_data[f"Stock_Lag_{lag}d"] = daily_data["Daily_Stock"].shift(lag)

    # Fill NaN values with 0 for lag features
    lag_columns = [col for col in daily_data.columns if "Lag" in col]
    daily_data[lag_columns] = daily_data[lag_columns].fillna(0)

    return daily_data


def create_seasonality_features(daily_data):
    """
    Create seasonality features based on historical patterns
    """
    # Add date components to the dataframe
    daily_data["DayOfWeek"] = daily_data.index.dayofweek
    daily_data["Month"] = daily_data.index.month
    daily_data["Quarter"] = daily_data.index.quarter

    # Calculate average sales by day of week
    day_of_week_avg = daily_data.groupby("DayOfWeek")["Daily_Sales"].transform("mean")
    daily_data["Sales_DayOfWeek_Avg"] = day_of_week_avg

    # Calculate average sales by month
    month_avg = daily_data.groupby("Month")["Daily_Sales"].transform("mean")
    daily_data["Sales_Month_Avg"] = month_avg

    # Calculate average sales by quarter
    quarter_avg = daily_data.groupby("Quarter")["Daily_Sales"].transform("mean")
    daily_data["Sales_Quarter_Avg"] = quarter_avg

    # Calculate ratio of current sales to average for that day of week
    daily_data["Sales_DayOfWeek_Ratio"] = daily_data["Daily_Sales"] / daily_data[
        "Sales_DayOfWeek_Avg"
    ].replace(0, 1)

    # Calculate ratio of current sales to average for that month
    daily_data["Sales_Month_Ratio"] = daily_data["Daily_Sales"] / daily_data[
        "Sales_Month_Avg"
    ].replace(0, 1)

    return daily_data


def create_expiration_risk_features(data):
    """
    Create features specifically for expiration risk prediction
    """
    print("Creating expiration risk features...")

    # Calculate the ratio of stock to sales velocity
    for window in [7, 14, 30]:
        # Days of stock based on recent sales velocity
        data[f"Days_of_Stock_{window}d"] = np.where(
            data[f"Sales_Velocity_{window}d"] > 0,
            data["Quantity in Stock (liters/kg)"] / data[f"Sales_Velocity_{window}d"],
            float("inf"),  # If velocity is 0, set to infinity
        )

        # Expiration risk based on days of stock vs days to expire
        data[f"Expiration_Risk_{window}d"] = np.where(
            data["Days_to_Expire"] < data[f"Days_of_Stock_{window}d"],
            1,  # Risk of expiration
            0,  # No risk
        )

        # Expiration risk score (0-100%)
        data[f"Expiration_Risk_Score_{window}d"] = np.where(
            data[f"Days_of_Stock_{window}d"] > 0,
            np.minimum(
                100, 100 * data["Days_to_Expire"] / data[f"Days_of_Stock_{window}d"]
            ),
            0,  # If days of stock is infinite (velocity = 0), set risk to 0
        )

        # Reverse the score so higher means more risk
        data[f"Expiration_Risk_Score_{window}d"] = (
            100 - data[f"Expiration_Risk_Score_{window}d"]
        )

        # Clip to 0-100 range
        data[f"Expiration_Risk_Score_{window}d"] = data[
            f"Expiration_Risk_Score_{window}d"
        ].clip(0, 100)

    # Create a combined expiration risk score (average of different windows)
    risk_score_columns = [
        col for col in data.columns if "Expiration_Risk_Score_" in col
    ]
    data["Expiration_Risk_Score"] = data[risk_score_columns].mean(axis=1)

    return data


def create_reorder_quantity_features(data):
    """
    Create features specifically for optimal reorder quantity prediction
    """
    print("Creating reorder quantity features...")

    # Calculate optimal reorder quantity based on different time windows
    for window in [7, 14, 30]:
        # Lead time (assumed to be 3 days, adjust as needed)
        lead_time = 3

        # Safety stock (based on sales volatility)
        data[f"Safety_Stock_{window}d"] = data[f"Sales_Volatility_{window}d"] * np.sqrt(
            lead_time
        )

        # Expected sales during lead time
        data[f"Lead_Time_Demand_{window}d"] = (
            data[f"Sales_Velocity_{window}d"] * lead_time
        )

        # Reorder point (when stock reaches this level, reorder)
        data[f"Reorder_Point_{window}d"] = (
            data[f"Lead_Time_Demand_{window}d"] + data[f"Safety_Stock_{window}d"]
        )

        # Economic order quantity (simplified version)
        # Assuming ordering cost is 100 and holding cost is 10% of item price
        ordering_cost = 100
        holding_cost_percent = 0.1

        data[f"Holding_Cost_{window}d"] = data["Price per Unit"] * holding_cost_percent

        # Annual demand based on recent velocity
        data[f"Annual_Demand_{window}d"] = data[f"Sales_Velocity_{window}d"] * 365

        # Economic Order Quantity formula
        data[f"EOQ_{window}d"] = np.sqrt(
            (2 * data[f"Annual_Demand_{window}d"] * ordering_cost)
            / data[f"Holding_Cost_{window}d"].replace(0, 0.01)
        )

        # Optimal reorder quantity (considering shelf life constraints)
        data[f"Optimal_Reorder_Quantity_{window}d"] = np.minimum(
            data[f"EOQ_{window}d"],
            data[f"Sales_Velocity_{window}d"] * data["Shelf Life (days)"],
        )

    # Create a combined optimal reorder quantity (weighted average of different windows)
    # Give more weight to more recent data
    data["Optimal_Reorder_Quantity"] = (
        0.5 * data["Optimal_Reorder_Quantity_7d"]
        + 0.3 * data["Optimal_Reorder_Quantity_14d"]
        + 0.2 * data["Optimal_Reorder_Quantity_30d"]
    )

    return data


def visualize_historical_features(
    data, output_dir="historical_diary_study/hist_dairy_plots"
):
    """
    Create visualizations of historical features
    """
    print("Creating visualizations of historical features...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Sample a few products for visualization
    sample_products = data["Product Name"].unique()[:3]

    for product in sample_products:
        product_data = data[data["Product Name"] == product].sort_values("Date_Sell")

        # Plot sales velocity over time with different windows
        plt.figure(figsize=(12, 6))
        plt.plot(
            product_data["Date_Sell"],
            product_data["Sales_Velocity"],
            label="Daily",
            alpha=0.5,
        )
        plt.plot(
            product_data["Date_Sell"], product_data["Sales_Velocity_7d"], label="7-day"
        )
        plt.plot(
            product_data["Date_Sell"],
            product_data["Sales_Velocity_30d"],
            label="30-day",
        )
        plt.title(f"Sales Velocity Over Time - {product}")
        plt.xlabel("Date")
        plt.ylabel("Sales Velocity (units/day)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/sales_velocity_{product.replace(' ', '_')}.png")
        plt.close()

        # Plot expiration risk score
        plt.figure(figsize=(12, 6))
        plt.plot(product_data["Date_Sell"], product_data["Expiration_Risk_Score"])
        plt.title(f"Expiration Risk Score Over Time - {product}")
        plt.xlabel("Date")
        plt.ylabel("Risk Score (0-100%)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/expiration_risk_{product.replace(' ', '_')}.png")
        plt.close()

        # Plot optimal reorder quantity
        plt.figure(figsize=(12, 6))
        plt.plot(product_data["Date_Sell"], product_data["Optimal_Reorder_Quantity"])
        plt.title(f"Optimal Reorder Quantity Over Time - {product}")
        plt.xlabel("Date")
        plt.ylabel("Quantity")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reorder_quantity_{product.replace(' ', '_')}.png")
        plt.close()

    # Plot distribution of expiration risk scores
    plt.figure(figsize=(10, 6))
    sns.histplot(data["Expiration_Risk_Score"], bins=20)
    plt.title("Distribution of Expiration Risk Scores")
    plt.xlabel("Risk Score (0-100%)")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_distribution.png")
    plt.close()

    # Plot distribution of optimal reorder quantities
    plt.figure(figsize=(10, 6))
    sns.histplot(data["Optimal_Reorder_Quantity"], bins=20)
    plt.title("Distribution of Optimal Reorder Quantities")
    plt.xlabel("Quantity")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_distribution.png")
    plt.close()

    # Plot correlation matrix of key features
    plt.figure(figsize=(16, 12))
    key_features = [
        "Sales_Velocity",
        "Sales_Velocity_7d",
        "Sales_Velocity_30d",
        "Sales_Volatility_7d",
        "Sales_Volatility_30d",
        "Expiration_Risk_Score",
        "Optimal_Reorder_Quantity",
        "Days_to_Expire",
        "Shelf Life (days)",
    ]
    correlation = data[key_features].corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix of Key Features")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/key_features_correlation.png")
    plt.close()


def main():
    """
    Main function to run the historical feature engineering process
    """
    # Create output directories
    os.makedirs("historical_diary_study/hist_dairy_plots", exist_ok=True)
    os.makedirs("historical_diary_study/hist_dairy_models", exist_ok=True)
    os.makedirs("historical_diary_study/hist_dairy_data", exist_ok=True)

    # Load data
    data = load_data("dairy_dataset.csv")

    # Create base features
    data = create_base_features(data)

    # Create historical features
    data = create_product_historical_features(data)

    # Create expiration risk features
    data = create_expiration_risk_features(data)

    # Create reorder quantity features
    data = create_reorder_quantity_features(data)

    # Visualize historical features
    visualize_historical_features(data)

    # Save the processed data
    data.to_csv(
        "historical_diary_study/hist_dairy_data/dairy_data_with_historical_features.csv",
        index=False,
    )

    print("Historical feature engineering completed successfully!")
    print(
        f"Processed data saved to hist_dairy_data/dairy_data_with_historical_features.csv"
    )
    print(f"Visualizations saved to hist_dairy_plots/")


if __name__ == "__main__":
    main()
