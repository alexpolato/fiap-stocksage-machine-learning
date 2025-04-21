import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pickle
import os

"""
This script implements a machine learning model to predict expiration risk for dairy products.
It uses the historical features created in the historical_features_implementation.py script.

The model predicts which products are at risk of expiring before being sold, allowing for
proactive measures like discounting or promotional activities.

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


def prepare_expiration_risk_data(data):
    """
    Prepare data specifically for expiration risk prediction
    """
    print("Preparing data for expiration risk prediction...")

    # Define target variable: Whether the product expired before being sold
    # We'll use the 'Expired' column that was created in the historical features script

    # If 'Expired' column doesn't exist, create it
    if "Expired" not in data.columns:
        # Calculate if product expired before being sold
        # This is a simplified version; the full implementation is in the historical features script
        data["Expired"] = 0

        # If expiration date is before sell date, mark as expired
        if "Expiration Date" in data.columns and "Date_Sell" in data.columns:
            data.loc[data["Expiration Date"] < data["Date_Sell"], "Expired"] = 1

        # If days to expire is less than days to sell, mark as expired
        elif "Days_to_Expire" in data.columns and "Days_to_Sell" in data.columns:
            data.loc[data["Days_to_Expire"] < data["Days_to_Sell"], "Expired"] = 1

    # Select features for expiration risk prediction
    risk_features = [
        # Product information
        "Product Name",
        "Shelf Life (days)",
        # Current state
        "Days_to_Expire",
        "Quantity in Stock (liters/kg)",
        # Historical sales patterns
        "Sales_Velocity",
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
        # Derived features
        "Days_of_Stock_7d",
        "Days_of_Stock_14d",
        "Days_of_Stock_30d",
        # Seasonality features
        "DayOfWeek",
        "Month",
        "Quarter",
        "Sales_DayOfWeek_Ratio",
        "Sales_Month_Ratio",
    ]

    # Filter features that exist in the dataset
    available_features = [f for f in risk_features if f in data.columns]

    # Create feature matrix and target vector
    X = data[available_features].copy()
    y = data["Expired"].copy()

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
        # Fill NaN values in target with 0 (assuming not expired is the safer default)
        y = y.fillna(0)

    return X, y


def train_expiration_risk_model(X, y):
    """
    Train a machine learning model to predict expiration risk
    """
    print("Training expiration risk prediction model...")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Create a pipeline with preprocessing and model
    # Include SimpleImputer to handle any remaining NaN values
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    # Define models to evaluate
    models = {
        "Random Forest": RandomForestClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(
            random_state=42, class_weight="balanced"
        ),
    }

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        print(f"  Training {name}...")

        # Update the classifier in the pipeline
        pipeline.set_params(classifier=model)

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        y_test_prob = pipeline.predict_proba(X_test)[:, 1]

        # Calculate metrics
        train_report = classification_report(y_train, y_train_pred, output_dict=True)
        test_report = classification_report(y_test, y_test_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_test, y_test_pred)

        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_test_prob)
        roc_auc = auc(fpr, tpr)

        # Store results
        results[name] = {
            "pipeline": pipeline,
            "train_report": train_report,
            "test_report": test_report,
            "conf_matrix": conf_matrix,
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc,
            "y_test": y_test,
            "y_test_pred": y_test_pred,
            "y_test_prob": y_test_prob,
        }

        # Print basic metrics
        print(f"    Accuracy (train): {train_report['accuracy']:.4f}")
        print(f"    Accuracy (test): {test_report['accuracy']:.4f}")
        print(f"    F1 Score (test): {test_report['1']['f1-score']:.4f}")
        print(f"    AUC: {roc_auc:.4f}")

    # Find the best model based on F1 score for the positive class (expired)
    best_model_name = max(
        results, key=lambda name: results[name]["test_report"]["1"]["f1-score"]
    )
    print(f"\nBest model: {best_model_name}")

    # Perform hyperparameter tuning for the best model
    print(f"Performing hyperparameter tuning for {best_model_name}...")

    if best_model_name == "Random Forest":
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [None, 10, 20, 30],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
        }
    elif best_model_name == "Gradient Boosting":
        param_grid = {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__learning_rate": [0.01, 0.05, 0.1],
            "classifier__max_depth": [3, 5, 7],
            "classifier__min_samples_split": [2, 5, 10],
        }
    else:  # Logistic Regression
        param_grid = {
            "classifier__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "classifier__solver": ["liblinear", "saga"],
            "classifier__penalty": ["l1", "l2"],
        }

    # Create a new pipeline for tuning
    tuning_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", models[best_model_name]),
        ]
    )

    # Perform grid search
    grid_search = GridSearchCV(
        tuning_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best F1 Score: {grid_search.best_score_:.4f}")

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions with the best model
    y_train_pred = best_model.predict(X_train)
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_test_pred)

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    roc_auc = auc(fpr, tpr)

    # Store results for the tuned model
    results["Tuned " + best_model_name] = {
        "pipeline": best_model,
        "train_report": train_report,
        "test_report": test_report,
        "conf_matrix": conf_matrix,
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
        "y_test_prob": y_test_prob,
    }

    # Print metrics for the tuned model
    print(f"Tuned model performance:")
    print(f"  Accuracy (train): {train_report['accuracy']:.4f}")
    print(f"  Accuracy (test): {test_report['accuracy']:.4f}")
    print(f"  F1 Score (test): {test_report['1']['f1-score']:.4f}")
    print(f"  AUC: {roc_auc:.4f}")

    # Set the final model
    final_model_name = "Tuned " + best_model_name
    final_model = results[final_model_name]["pipeline"]

    return final_model, results, X_test, y_test


def visualize_model_results(results, X_test, output_dir="dairy_plots"):
    """
    Create visualizations of model results
    """
    print("Creating visualizations of model results...")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compare model performance
    model_names = list(results.keys())
    accuracy = [results[name]["test_report"]["accuracy"] for name in model_names]
    f1_score = [results[name]["test_report"]["1"]["f1-score"] for name in model_names]
    auc_score = [results[name]["roc_auc"] for name in model_names]

    # Create comparison dataframe
    comparison_df = pd.DataFrame(
        {
            "Model": model_names,
            "Accuracy": accuracy,
            "F1 Score": f1_score,
            "AUC": auc_score,
        }
    )

    # Save comparison to CSV
    comparison_df.to_csv(
        f"{output_dir}/expiration_risk_model_comparison.csv", index=False
    )

    # Plot model comparison - F1 Score
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="F1 Score", data=comparison_df)
    plt.title("Expiration Risk Model Comparison - F1 Score (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_model_comparison_f1.png")
    plt.close()

    # Plot model comparison - AUC
    plt.figure(figsize=(12, 6))
    sns.barplot(x="Model", y="AUC", data=comparison_df)
    plt.title("Expiration Risk Model Comparison - AUC (higher is better)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_model_comparison_auc.png")
    plt.close()

    # Plot ROC curves for all models
    plt.figure(figsize=(10, 8))

    for name in model_names:
        plt.plot(
            results[name]["fpr"],
            results[name]["tpr"],
            label=f"{name} (AUC = {results[name]['roc_auc']:.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Expiration Risk Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_roc_curves.png")
    plt.close()

    # Plot confusion matrix for the best model
    best_model_name = model_names[
        -1
    ]  # Assuming the last model is the best (tuned) model
    conf_matrix = results[best_model_name]["conf_matrix"]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Expired", "Expired"],
        yticklabels=["Not Expired", "Expired"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/expiration_risk_confusion_matrix.png")
    plt.close()

    # If the best model is a tree-based model, plot feature importances
    best_model = results[best_model_name]["pipeline"]
    if hasattr(best_model.named_steps["classifier"], "feature_importances_"):
        # Get feature importances
        importances = best_model.named_steps["classifier"].feature_importances_

        # Get feature names from X_test
        feature_names = X_test.columns

        # Create a dataframe of feature importances
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importances}
        ).sort_values("Importance", ascending=False)

        # Save feature importances to CSV
        importance_df.to_csv(
            f"{output_dir}/expiration_risk_feature_importances.csv", index=False
        )

        # Plot top 20 feature importances
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        sns.barplot(x="Importance", y="Feature", data=top_features)
        plt.title(f"Top 20 Feature Importances - {best_model_name}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/expiration_risk_feature_importances.png")
        plt.close()


def save_model(
    model,
    file_path="historical_diary_study/hist_dairy_models/expiration_risk_model.pkl",
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
    Main function to run the expiration risk prediction model
    """
    # Create output directories
    os.makedirs("historical_diary_study/hist_dairy_plots", exist_ok=True)
    os.makedirs("historical_diary_study/hist_dairy_models", exist_ok=True)

    # Load data with historical features
    data = load_data()

    # Prepare data for expiration risk prediction
    X, y = prepare_expiration_risk_data(data)

    # Train the model
    model, results, X_test, y_test = train_expiration_risk_model(X, y)

    # Visualize model results
    visualize_model_results(results, X_test)

    # Save the model
    save_model(model)

    print("Expiration risk prediction model training completed successfully!")
    print(f"Model saved to dairy_models/expiration_risk_model.pkl")
    print(f"Visualizations saved to dairy_plots/")


if __name__ == "__main__":
    main()
