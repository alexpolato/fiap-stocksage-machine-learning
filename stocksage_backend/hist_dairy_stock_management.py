import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

"""
This script integrates the historical feature engineering and prediction models
into a unified workflow for dairy stock management.

It combines:
1. Historical feature engineering
2. Expiration risk prediction
3. Optimal reorder quantity prediction

This script serves as the main entry point for the complete system.
"""


def load_data(file_path="dairy_dataset.csv"):
    """
    Load the original dairy dataset
    """
    print("Loading dataset...")
    data = pd.read_csv(file_path)

    # Convert date columns to datetime
    date_columns = ["Date", "Production Date", "Expiration Date"]
    for col in date_columns:
        data[col] = pd.to_datetime(data[col])

    return data


def process_historical_features(data):
    """
    Process historical features using the implementation from historical_features_implementation.py
    """
    print("Processing historical features...")

    # Import functions from historical_features_implementation.py
    from historical_features_implementation import (
        create_base_features,
        create_product_historical_features,
        create_expiration_risk_features,
        create_reorder_quantity_features,
    )

    # Create base features
    data = create_base_features(data)

    # Create historical features
    data = create_product_historical_features(data)

    # Create expiration risk features
    data = create_expiration_risk_features(data)

    # Create reorder quantity features
    data = create_reorder_quantity_features(data)

    return data


def predict_expiration_risk(
    data,
    model_path="historical_diary_study/hist_dairy_models/expiration_risk_model.pkl",
):
    """
    Predict expiration risk using the trained model
    """
    print("Predicting expiration risk...")

    # Check if model exists, if not, train it
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training a new model...")
        from expiration_risk_model import (
            prepare_expiration_risk_data,
            train_expiration_risk_model,
        )

        # Prepare data
        X, y = prepare_expiration_risk_data(data)

        # Train model
        model, _, _, _ = train_expiration_risk_model(X, y)

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Import prepare_expiration_risk_data function
        from expiration_risk_model import prepare_expiration_risk_data

        # Prepare data
        X, _ = prepare_expiration_risk_data(data)

    # Make predictions
    risk_probabilities = model.predict_proba(X)[
        :, 1
    ]  # Probability of class 1 (expiration)

    # Add predictions to the data
    data["Expiration_Risk_Probability"] = risk_probabilities

    # Classify high-risk products (probability > 0.5)
    data["High_Expiration_Risk"] = (data["Expiration_Risk_Probability"] > 0.5).astype(
        int
    )

    return data


def predict_reorder_quantity(
    data,
    model_path="historical_diary_study/hist_dairy_models/reorder_quantity_model.pkl",
):
    """
    Predict optimal reorder quantity using the trained model
    """
    print("Predicting optimal reorder quantities...")

    # Check if model exists, if not, train it
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training a new model...")
        from reorder_quantity_model import (
            prepare_reorder_quantity_data,
            train_reorder_quantity_model,
        )

        # Prepare data
        X, y = prepare_reorder_quantity_data(data)

        # Train model
        model, _, _, _ = train_reorder_quantity_model(X, y)

        # Save model
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    else:
        # Load the model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Import prepare_reorder_quantity_data function
        from reorder_quantity_model import prepare_reorder_quantity_data

        # Prepare data
        X, _ = prepare_reorder_quantity_data(data)

    # Make predictions
    predicted_quantities = model.predict(X)

    # Add predictions to the data
    data["Predicted_Reorder_Quantity"] = predicted_quantities

    # Ensure predictions are non-negative
    data["Predicted_Reorder_Quantity"] = np.maximum(
        0, data["Predicted_Reorder_Quantity"]
    )

    return data


def generate_stock_management_recommendations(data):
    """
    Generate actionable recommendations based on predictions
    """
    print("Generating stock management recommendations...")

    # Create a recommendations dataframe
    recommendations = pd.DataFrame(
        {
            "Product Name": data["Product Name"],
            "Date": data["Date"],
            "Current Stock": data["Quantity in Stock (liters/kg)"],
            "Expiration Date": data["Expiration Date"],
            "Days to Expire": data["Days_to_Expire"],
            "Sales Velocity (7d)": data["Sales_Velocity_7d"],
            "Expiration Risk": data["Expiration_Risk_Probability"],
            "Recommended Reorder Quantity": data["Predicted_Reorder_Quantity"],
        }
    )

    # Add recommendation type
    conditions = [
        (data["High_Expiration_Risk"] == 1)
        & (data["Quantity in Stock (liters/kg)"] > 0),
        (
            data["Quantity in Stock (liters/kg)"]
            <= data["Minimum Stock Threshold (liters/kg)"]
        ),
        (data["Quantity in Stock (liters/kg)"] > 0),
    ]

    choices = [
        "URGENT: High risk of expiration - Consider discounting",
        "REORDER: Stock below threshold - Place order",
        "MONITOR: Normal stock levels",
    ]

    recommendations["Recommendation"] = np.select(
        conditions, choices, default="MONITOR: Normal stock levels"
    )

    # Add price recommendation for high-risk products
    discount_conditions = [
        (recommendations["Recommendation"].str.contains("URGENT")),
        (recommendations["Recommendation"].str.contains("MONITOR")),
    ]

    # Calculate recommended discount based on expiration risk
    # Higher risk = higher discount
    base_price = data["Price per Unit"]
    max_discount = 0.3  # Maximum 30% discount

    discount_rate = np.minimum(
        max_discount, data["Expiration_Risk_Probability"] * max_discount
    )
    recommended_price = base_price * (1 - discount_rate)

    discount_choices = [recommended_price, base_price]

    recommendations["Recommended Price"] = np.select(
        discount_conditions, discount_choices, default=base_price
    )

    # Calculate potential revenue impact
    recommendations["Current Expected Revenue"] = (
        data["Quantity in Stock (liters/kg)"] * data["Price per Unit (sold)"]
    )
    recommendations["Recommended Revenue"] = (
        data["Quantity in Stock (liters/kg)"] * recommendations["Recommended Price"]
    )
    recommendations["Revenue Impact"] = (
        recommendations["Recommended Revenue"]
        - recommendations["Current Expected Revenue"]
    )

    # Sort by recommendation priority (URGENT first, then REORDER, then MONITOR)
    recommendations["Priority"] = recommendations["Recommendation"].apply(
        lambda x: 0 if "URGENT" in x else (1 if "REORDER" in x else 2)
    )

    recommendations = recommendations.sort_values(
        ["Priority", "Expiration Risk"], ascending=[True, False]
    )

    # Drop the priority column as it's just for sorting
    recommendations = recommendations.drop("Priority", axis=1)

    return recommendations


def visualize_recommendations(recommendations, output_dir="dairy_plots"):
    """
    Create visualizations of stock management recommendations
    """
    print("Creating visualizations of recommendations...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Count recommendations by type
    rec_counts = recommendations["Recommendation"].value_counts()

    # Plot recommendation distribution
    plt.figure(figsize=(10, 6))
    sns.barplot(x=rec_counts.index, y=rec_counts.values)
    plt.title("Distribution of Stock Management Recommendations")
    plt.xlabel("Recommendation Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recommendation_distribution.png")
    plt.close()

    # Plot expiration risk by product
    plt.figure(figsize=(12, 6))
    product_risk = (
        recommendations.groupby("Product Name")["Expiration Risk"]
        .mean()
        .sort_values(ascending=False)
    )
    sns.barplot(x=product_risk.index, y=product_risk.values)
    plt.title("Average Expiration Risk by Product")
    plt.xlabel("Product")
    plt.ylabel("Average Risk (0-1)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_by_product.png")
    plt.close()

    # Plot recommended reorder quantities by product
    plt.figure(figsize=(12, 6))
    reorder_qty = (
        recommendations.groupby("Product Name")["Recommended Reorder Quantity"]
        .mean()
        .sort_values(ascending=False)
    )
    sns.barplot(x=reorder_qty.index, y=reorder_qty.values)
    plt.title("Average Recommended Reorder Quantity by Product")
    plt.xlabel("Product")
    plt.ylabel("Quantity")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_by_product.png")
    plt.close()

    # Plot revenue impact by recommendation type
    plt.figure(figsize=(10, 6))
    impact_by_rec = recommendations.groupby("Recommendation")["Revenue Impact"].sum()
    sns.barplot(x=impact_by_rec.index, y=impact_by_rec.values)
    plt.title("Total Revenue Impact by Recommendation Type")
    plt.xlabel("Recommendation Type")
    plt.ylabel("Revenue Impact")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/revenue_impact_by_recommendation.png")
    plt.close()


def main():
    """
    Main function to run the integrated dairy stock management system
    """
    # Create output directories
    os.makedirs("dairy_plots", exist_ok=True)
    os.makedirs("dairy_models", exist_ok=True)
    os.makedirs("dairy_data", exist_ok=True)
    os.makedirs("dairy_recommendations", exist_ok=True)

    # Load data
    data = load_data("dairy_dataset.csv")

    # Process historical features
    data_with_features = process_historical_features(data)

    # Save processed data
    data_with_features.to_csv("dairy_data/dairy_data_with_features.csv", index=False)

    # Predict expiration risk
    data_with_risk = predict_expiration_risk(data_with_features)

    # Predict optimal reorder quantity
    data_with_predictions = predict_reorder_quantity(data_with_risk)

    # Generate recommendations
    recommendations = generate_stock_management_recommendations(data_with_predictions)

    # Save recommendations
    current_date = datetime.now().strftime("%Y-%m-%d")
    recommendations.to_csv(
        f"dairy_recommendations/stock_recommendations_{current_date}.csv", index=False
    )

    # Visualize recommendations
    visualize_recommendations(recommendations)

    # Print summary
    urgent_count = recommendations[
        recommendations["Recommendation"].str.contains("URGENT")
    ].shape[0]
    reorder_count = recommendations[
        recommendations["Recommendation"].str.contains("REORDER")
    ].shape[0]
    monitor_count = recommendations[
        recommendations["Recommendation"].str.contains("MONITOR")
    ].shape[0]

    print("\nStock Management Summary:")
    print(f"  URGENT actions needed: {urgent_count}")
    print(f"  Products to REORDER: {reorder_count}")
    print(f"  Products to MONITOR: {monitor_count}")

    total_revenue_impact = recommendations["Revenue Impact"].sum()
    print(f"\nTotal Revenue Impact: ${total_revenue_impact:.2f}")

    print(
        f"\nDetailed recommendations saved to: dairy_recommendations/stock_recommendations_{current_date}.csv"
    )
    print(f"Visualizations saved to: dairy_plots/")


if __name__ == "__main__":
    main()
