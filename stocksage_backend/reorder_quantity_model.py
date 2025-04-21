import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os

"""
This script implements a machine learning model to predict optimal reorder quantities for dairy products.
It uses the historical features created in the historical_features_implementation.py script.

The model predicts how much of each product should be reordered based on historical sales patterns,
shelf life constraints, and current inventory levels.

Updated to handle NaN values robustly.
"""


def load_data(
    file_path="historical_diary_study/hist_dairy_data/dairy_data_with_historical_features.csv",
):
    """
    Load the processed data with historical features
    """
    print("Loading data with historical features...")
    data = pd.read_csv(file_path)

    # Convert date columns to datetime
    date_columns = ["Date", "Production Date", "Expiration Date", "Date_Sell"]
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col])

    return data


def prepare_reorder_quantity_data(data):
    """
    Prepare data specifically for optimal reorder quantity prediction
    """
    print("Preparing data for optimal reorder quantity prediction...")

    # Define target variable: Optimal reorder quantity
    # We'll use the 'Optimal_Reorder_Quantity' column that was created in the historical features script

    # If 'Optimal_Reorder_Quantity' column doesn't exist, create it
    if "Optimal_Reorder_Quantity" not in data.columns:
        # Calculate a simple version of optimal reorder quantity
        # This is a simplified version; the full implementation is in the historical features script
        lead_time = 3  # Assumed lead time in days
        safety_stock_factor = 1.5  # Safety stock multiplier

        # Calculate safety stock based on sales volatility
        data["Safety_Stock"] = (
            data["Sales_Velocity_7d"].rolling(window=7).std().fillna(0)
            * safety_stock_factor
        )

        # Calculate lead time demand
        data["Lead_Time_Demand"] = data["Sales_Velocity_7d"] * lead_time

        # Calculate optimal reorder quantity
        data["Optimal_Reorder_Quantity"] = (
            data["Lead_Time_Demand"] + data["Safety_Stock"]
        )

        # Ensure reorder quantity doesn't exceed what can be sold before expiration
        data["Optimal_Reorder_Quantity"] = np.minimum(
            data["Optimal_Reorder_Quantity"],
            data["Sales_Velocity_7d"] * data["Shelf Life (days)"],
        )

    # Select features for reorder quantity prediction
    reorder_features = [
        # Product information
        "Product Name",
        "Shelf Life (days)",
        # Current state
        "Quantity in Stock (liters/kg)",
        "Minimum Stock Threshold (liters/kg)",
        # Historical sales patterns
        "Sales_Velocity_7d",
        "Sales_Velocity_14d",
        "Sales_Velocity_30d",
        "Sales_Volatility_7d",
        "Sales_Volatility_14d",
        "Sales_Volatility_30d",
        # Exponential weighted features
        "Sales_Velocity_EWM_7d",
        "Sales_Velocity_EWM_14d",
        "Sales_Velocity_EWM_30d",
        # Seasonality features
        "Sales_DayOfWeek_Avg",
        "Sales_Month_Avg",
        "Sales_DayOfWeek_Ratio",
        "Sales_Month_Ratio",
        # Time-based features
        "DayOfWeek",
        "Month",
        "Quarter",
        # Reorder-specific features
        "Lead_Time_Demand_7d",
        "Lead_Time_Demand_14d",
        "Lead_Time_Demand_30d",
        "Safety_Stock_7d",
        "Safety_Stock_14d",
        "Safety_Stock_30d",
    ]

    # Filter features that exist in the dataset
    available_features = [f for f in reorder_features if f in data.columns]

    # Create feature matrix and target vector
    X = data[available_features].copy()
    y = data["Optimal_Reorder_Quantity"].copy()

    # Handle NaN values in X
    print(f"Before handling NaN values: {X.isna().sum().sum()} NaN values in X")

    # Option 1: Drop rows with NaN values
    # X_clean = X.dropna()
    # y_clean = y[X_clean.index]

    # Option 2: Fill NaN values with appropriate values (better approach)
    # For numeric columns, fill with median
    numeric_cols = X.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # For categorical columns, fill with mode
    categorical_cols = X.select_dtypes(exclude=["number"]).columns
    for col in categorical_cols:
        X[col] = X[col].fillna(
            X[col].mode()[0] if not X[col].mode().empty else "Unknown"
        )

    print(f"After handling NaN values: {X.isna().sum().sum()} NaN values in X")

    # Handle categorical features
    X = pd.get_dummies(X, columns=["Product Name"], drop_first=True)

    # Handle NaN values in y
    if y.isna().any():
        print(f"Found {y.isna().sum()} NaN values in target variable")
        # Fill NaN values in target with median
        y = y.fillna(y.median())

    return X, y


def train_reorder_quantity_model(X, y):
    """
    Train a machine learning model to predict optimal reorder quantities
    """
    print("Training optimal reorder quantity prediction model...")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create a pipeline with preprocessing and model
    # Include SimpleImputer to handle any remaining NaN values
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(random_state=42)),
        ]
    )

    # Define models to evaluate
    models = {
        "Random Forest": RandomForestRegressor(random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Ridge Regression": Ridge(random_state=42),
        "Lasso Regression": Lasso(random_state=42),
        "ElasticNet": ElasticNet(random_state=42),
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"  Training {name}...")

        # Update the regressor in the pipeline
        pipeline.set_params(regressor=model)

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)

        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        # Store results
        results[name] = {
            "pipeline": pipeline,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
        }

        # Print basic metrics
        print(f"    RMSE (train): {train_rmse:.2f}")
        print(f"    RMSE (test): {test_rmse:.2f}")
        print(f"    MAE (test): {test_mae:.2f}")
        print(f"    R² (test): {test_r2:.4f}")

    # Find the best model based on test RMSE
    best_model_name = min(results, key=lambda name: results[name]["test_rmse"])
    print(f"\nBest model: {best_model_name}")

    # Perform hyperparameter tuning for the best model
    print(f"Performing hyperparameter tuning for {best_model_name}...")

    if best_model_name == "Random Forest":
        param_grid = {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [None, 10, 20, 30],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            "regressor__n_estimators": [100, 200, 300],
            "regressor__learning_rate": [0.01, 0.05, 0.1],
            "regressor__max_depth": [3, 5, 7],
            "regressor__min_samples_split": [2, 5, 10],
        }
    elif best_model_name == "Ridge Regression":
        param_grid = {
            "regressor__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
            "regressor__solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
        }
    elif best_model_name == "Lasso Regression":
        param_grid = {
            "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            "regressor__selection": ["cyclic", "random"],
        }
    else:  # ElasticNet
        param_grid = {
            "regressor__alpha": [0.001, 0.01, 0.1, 1.0],
            "regressor__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        }

    # Create a new pipeline for tuning
    tuning_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("regressor", models[best_model_name]),
        ]
    )

    # Perform grid search
    grid_search = GridSearchCV(
        tuning_pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {-grid_search.best_score_:.2f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Store results for the tuned model
    results["Tuned " + best_model_name] = {
        "pipeline": best_model,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }

    # Print metrics for the tuned model
    print(f"Tuned model performance:")
    print(f"  RMSE (train): {train_rmse:.2f}")
    print(f"  RMSE (test): {test_rmse:.2f}")
    print(f"  MAE (test): {test_mae:.2f}")
    print(f"  R² (test): {test_r2:.4f}")

    # Set the final model
    final_model_name = "Tuned " + best_model_name
    final_model = results[final_model_name]["pipeline"]

    return final_model, results, X_test, y_test


def visualize_model_results(
    results, X_test, output_dir="historical_diary_study/hist_dairy_plots"
):
    """
    Create visualizations of model results
    """
    print("Creating visualizations of model results...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compare model performance
    model_names = list(results.keys())
    rmse = [results[name]["test_rmse"] for name in model_names]
    mae = [results[name]["test_mae"] for name in model_names]
    r2 = [results[name]["test_r2"] for name in model_names]

    # Create comparison dataframe
    comparison_df = pd.DataFrame(
        {"Model": model_names, "RMSE": rmse, "MAE": mae, "R² Score": r2}
    )

    # Save comparison to CSV
    comparison_df.to_csv(
        f"{output_dir}/reorder_quantity_model_comparison.csv", index=False
    )

    # Plot model comparison - RMSE
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="RMSE", data=comparison_df)
    plt.title("Reorder Quantity Model Comparison - RMSE (lower is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_model_comparison_rmse.png")
    plt.close()

    # Plot model comparison - R²
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="R² Score", data=comparison_df)
    plt.title("Reorder Quantity Model Comparison - R² Score (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_model_comparison_r2.png")
    plt.close()

    # Plot actual vs predicted values for the best model
    best_model_name = model_names[
        -1
    ]  # Assuming the last model is the best (tuned) model
    y_test = results[best_model_name]["y_test"]
    y_test_pred = results[best_model_name]["y_test_pred"]

    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel("Actual Reorder Quantity")
    plt.ylabel("Predicted Reorder Quantity")
    plt.title(f"Actual vs Predicted Reorder Quantity - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_actual_vs_predicted.png")
    plt.close()

    # Plot prediction error distribution
    error = y_test_pred - y_test

    plt.figure(figsize=(10, 6))
    sns.histplot(error, kde=True)
    plt.xlabel("Prediction Error")
    plt.title("Distribution of Prediction Errors")
    plt.axvline(x=0, color="r", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reorder_quantity_error_distribution.png")
    plt.close()

    # If the best model is a tree-based model, plot feature importances
    best_model = results[best_model_name]["pipeline"]
    if hasattr(best_model.named_steps["regressor"], "feature_importances_"):
        # Get feature importances
        importances = best_model.named_steps["regressor"].feature_importances_

        # Get feature names from X_test
        feature_names = X_test.columns

        # Create a dataframe of feature importances
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        # Save feature importances to CSV
        importance_df.to_csv(
            f"{output_dir}/reorder_quantity_feature_importances.csv", index=False
        )

        # Plot top 20 feature importances
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(x="Importance", y="Feature", data=top_features)
        plt.title(f"Top 20 Feature Importances - {best_model_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/reorder_quantity_feature_importances.png")
        plt.close()


def save_model(
    model,
    file_path="historical_diary_study/hist_dairy_models/reorder_quantity_model.pkl",
):
    """
    Save the trained model to a file
    """
    print(f"Saving model to {file_path}...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(model, f)


def main():
    """
    Main function to run the optimal reorder quantity prediction model
    """
    # Create output directories
    os.makedirs("historical_diary_study/hist_dairy_plots", exist_ok=True)
    os.makedirs("historical_diary_study/hist_dairy_models", exist_ok=True)

    # Load data with historical features
    data = load_data()

    # Prepare data for reorder quantity prediction
    X, y = prepare_reorder_quantity_data(data)

    # Train the model
    model, results, X_test, y_test = train_reorder_quantity_model(X, y)

    # Visualize model results
    visualize_model_results(results, X_test)

    # Save the model
    save_model(model)

    print("Optimal reorder quantity prediction model training completed successfully!")
    print(f"Model saved to hist_dairy_models/reorder_quantity_model.pkl")
    print(f"Visualizations saved to hist_dairy_plots/")


if __name__ == "__main__":
    main()
