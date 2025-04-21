import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import time

# Create directory for model evaluation results
os.makedirs("dairy/dairy_evaluation", exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
X_train = np.load("dairy/dairy_data/X_train_processed.npy")
X_test = np.load("dairy/dairy_data/X_test_processed.npy")
y_train = np.load("dairy/dairy_data/y_train.npy")
y_test = np.load("dairy/dairy_data/y_test.npy")

# Load feature information
with open("dairy/dairy_data/feature_lists.pkl", "rb") as f:
    feature_lists = pickle.load(f)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Define models to train
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}

# Dictionary to store results
results = {}
trained_models = {}


# Function to evaluate model
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train the model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "training_time": training_time,
        "y_test_pred": y_test_pred,
    }


# Train and evaluate each model
print("Training and evaluating models...")
for name, model in models.items():
    print(f"Training {name}...")
    model_results = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = model_results
    trained_models[name] = model

    print(f"  RMSE (test): {model_results['test_rmse']:.2f}")
    print(f"  MAE (test): {model_results['test_mae']:.2f}")
    print(f"  R² (test): {model_results['test_r2']:.4f}")
    print(f"  Training time: {model_results['training_time']:.2f} seconds")

# Create results dataframe
results_df = pd.DataFrame(
    {
        "Model": list(results.keys()),
        "RMSE (Train)": [results[model]["train_rmse"] for model in results],
        "RMSE (Test)": [results[model]["test_rmse"] for model in results],
        "MAE (Train)": [results[model]["train_mae"] for model in results],
        "MAE (Test)": [results[model]["test_mae"] for model in results],
        "R² (Train)": [results[model]["train_r2"] for model in results],
        "R² (Test)": [results[model]["test_r2"] for model in results],
        "Training Time (s)": [results[model]["training_time"] for model in results],
    }
)

# Save results
results_df.to_csv("dairy/dairy_evaluation/model_comparison.csv", index=False)
print("\nModel comparison results saved to 'dairy_evaluation/model_comparison.csv'")

# Visualize model performance
plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="RMSE (Test)", data=results_df)
plt.title("Model Comparison - RMSE (Test)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_evaluation/model_comparison_rmse.png")

plt.figure(figsize=(12, 6))
sns.barplot(x="Model", y="R² (Test)", data=results_df)
plt.title("Model Comparison - R² (Test)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_evaluation/model_comparison_r2.png")

# Find the best model based on test RMSE
best_model_name = results_df.loc[results_df["RMSE (Test)"].idxmin(), "Model"]
print(f"\nBest model based on RMSE: {best_model_name}")

# Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == "Random Forest":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }
    base_model = RandomForestRegressor(random_state=42)
elif best_model_name == "Gradient Boosting":
    param_grid = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "min_samples_split": [2, 5, 10],
    }
    base_model = GradientBoostingRegressor(random_state=42)
elif best_model_name == "SVR":
    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01],
        "kernel": ["rbf", "linear"],
    }
    base_model = SVR()
elif best_model_name == "KNN":
    param_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    base_model = KNeighborsRegressor()
elif best_model_name == "Ridge Regression":
    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg"],
    }
    base_model = Ridge()
elif best_model_name == "Lasso Regression":
    param_grid = {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
        "selection": ["cyclic", "random"],
    }
    base_model = Lasso()
else:  # Linear Regression
    print("Linear Regression doesn't have hyperparameters to tune. Skipping tuning.")
    tuned_model = trained_models[best_model_name]
    param_grid = {}

# Perform grid search if there are hyperparameters to tune
if param_grid:
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best RMSE: {-grid_search.best_score_:.2f}")

    # Save the best model
    tuned_model = grid_search.best_estimator_

    # Evaluate tuned model
    tuned_results = evaluate_model(tuned_model, X_train, X_test, y_train, y_test)

    print(f"Tuned model performance:")
    print(f"  RMSE (test): {tuned_results['test_rmse']:.2f}")
    print(f"  MAE (test): {tuned_results['test_mae']:.2f}")
    print(f"  R² (test): {tuned_results['test_r2']:.4f}")

    # Add tuned model to results
    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                {
                    "Model": [f"{best_model_name} (Tuned)"],
                    "RMSE (Train)": [tuned_results["train_rmse"]],
                    "RMSE (Test)": [tuned_results["test_rmse"]],
                    "MAE (Train)": [tuned_results["train_mae"]],
                    "MAE (Test)": [tuned_results["test_mae"]],
                    "R² (Train)": [tuned_results["train_r2"]],
                    "R² (Test)": [tuned_results["test_r2"]],
                    "Training Time (s)": [tuned_results["training_time"]],
                }
            ),
        ]
    )

    # Update results file
    results_df.to_csv("dairy/dairy_evaluation/model_comparison.csv", index=False)

    # Save tuned model predictions for analysis
    final_model = tuned_model
    final_model_name = f"{best_model_name} (Tuned)"
else:
    final_model = trained_models[best_model_name]
    final_model_name = best_model_name

# Save the final model
print(f"\nSaving the final model: {final_model_name}")
with open("dairy/dairy_models/final_model.pkl", "wb") as f:
    pickle.dump(final_model, f)

# Load the original test data for analysis
test_data_original = pd.read_csv("dairy/dairy_data/test_data.csv")

# Add predictions to the test data
test_data_original["Predicted_Real_Revenue"] = (
    results[best_model_name]["y_test_pred"]
    if param_grid == {}
    else tuned_results["y_test_pred"]
)
test_data_original["Prediction_Error"] = (
    test_data_original["Predicted_Real_Revenue"]
    - test_data_original["Quantity Sold (liters/kg)"]
)
test_data_original["Absolute_Error"] = abs(test_data_original["Prediction_Error"])

# Save the test data with predictions
test_data_original.to_csv(
    "dairy_evaluation/test_data_with_predictions.csv", index=False
)

# Visualize actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(
    test_data_original["Real_Revenue"],
    test_data_original["Predicted_Real_Revenue"],
    alpha=0.5,
)
plt.plot(
    [0, test_data_original["Real_Revenue"].max()],
    [0, test_data_original["Real_Revenue"].max()],
    "r--",
)
plt.xlabel("Actual Real_Revenue")
plt.ylabel("Predicted Real_Revenue")
plt.title(f"Actual vs Predicted Real_Revenue - {final_model_name}")
plt.tight_layout()
plt.savefig("dairy/dairy_evaluation/actual_vs_predicted.png")

# Analyze prediction errors by product
plt.figure(figsize=(12, 6))
error_by_product = (
    test_data_original.groupby("Product Name")["Absolute_Error"]
    .mean()
    .sort_values(ascending=False)
)
sns.barplot(x=error_by_product.index, y=error_by_product.values)
plt.title("Mean Absolute Error by Product")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("dairy/dairy_evaluation/error_by_product.png")


# Feature importance analysis (if applicable)
if hasattr(final_model, "feature_importances_"):
    # Load the preprocessor to get feature names
    with open("dairy/dairy_models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    # Get feature names
    categorical_features = feature_lists["categorical_features"]
    numerical_features = feature_lists["numerical_features"]

    # Get one-hot encoded feature names
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()

    # Combine all feature names
    all_feature_names = numerical_features + cat_feature_names

    # Get feature importances
    feature_importances = final_model.feature_importances_

    # Create a dataframe of feature importances
    importance_df = pd.DataFrame(
        {"Feature": all_feature_names, "Importance": feature_importances}
    ).sort_values("Importance", ascending=False)

    # Save feature importances
    importance_df.to_csv("dairy/dairy_evaluation/feature_importances.csv", index=False)

    # Visualize top 20 feature importances
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(20)
    sns.barplot(x="Importance", y="Feature", data=top_features)
    plt.title(f"Top 20 Feature Importances - {final_model_name}")
    plt.tight_layout()
    plt.savefig("dairy/dairy_evaluation/feature_importances.png")

    print("\nFeature importance analysis completed and saved.")

print("\nModel training and evaluation completed successfully!")
