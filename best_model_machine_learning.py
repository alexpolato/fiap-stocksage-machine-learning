import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore

# Import necessary libraries for model selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    StackingRegressor,
    BaggingRegressor,
    AdaBoostRegressor,
    VotingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings


#### Seleção do modelo para prever a colunas Approx. Total Revenue(INR)
def treat_outliers(df, col):
    z_scores = zscore(df[col])
    df_no_outliers = df[(z_scores < 3)]
    return df_no_outliers


def machine_leaning_best_model(df):
    # Remover outliers
    df = treat_outliers(df, "Approx. Total Revenue(INR)")
    # Preparando os dados para modelagem
    X = df.drop(
        columns=[
            "Approx. Total Revenue(INR)",
            "Price per Unit (sold)",
            "Quantity Sold (liters/kg)",
        ]
    )
    y = df["Approx. Total Revenue(INR)"]

    # Dividindo em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = [
        "Location",
        "Farm Size",
        "Product Name",
        "Brand",
        "Customer Location",
        "Sales Channel",
    ]
    numerical_cols = [
        "Total Land Area (acres)",
        "Number of Cows",
        "Quantity (liters/kg)",
        "Price per Unit",
        "Total Value",
        "Shelf Life (days)",
        "Quantity in Stock (liters/kg)",
        "Minimum Stock Threshold (liters/kg)",
        "Reorder Quantity (liters/kg)",
        "Days Before Expire",
        "Days to Sell",
    ]

    categorical_features = [
        "Location",
        "Product Name",
        "Brand",
        "Storage Condition",
        "Customer Location",
        "Sales Channel",
    ]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), categorical_features)],
        remainder="passthrough",
    )

    cat_preprocessor = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_preprocessor = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_preprocessor, categorical_cols),
            ("num", num_preprocessor, numerical_cols),
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    scaler = StandardScaler()
    y_train_normalized = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_normalized = scaler.transform(y_test.values.reshape(-1, 1))

    base_models = [
        ("linear_regression", LinearRegression()),
        ("decision_tree", DecisionTreeRegressor()),
        ("knn", KNeighborsRegressor()),
    ]

    # Define the stacking ensemble model
    stacking_model = StackingRegressor(
        estimators=base_models, final_estimator=LinearRegression()
    )

    # Define the bagging ensemble model
    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor()
    )  # Corrigido: usar 'estimator' em vez de 'base_estimator'

    # Definir o modelo de ensemble boosting
    boosting_model = AdaBoostRegressor(estimator=DecisionTreeRegressor())

    # Define the voting ensemble model
    voting_model = VotingRegressor(estimators=base_models)

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("KNN", KNeighborsRegressor()),
        ("Stacking", stacking_model),
        ("Bagging", bagging_model),
        ("Boosting", boosting_model),
        ("Voting", voting_model),
    ]

    k = 5  # Number of folds
    mse_scores = []

    for model_name, model in models:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(
            model,
            X_train_preprocessed,
            y_train_normalized.ravel(),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        mse_scores.append((-1) * scores)
        mean_mse = np.mean((-1) * scores)

        print(f"{model_name} - Mean MSE: {mean_mse:.4f}")

    # Hyperparameter grid for Linear Regression
    linear_regression_params = {
        "fit_intercept": [True, False],
    }

    # Hyperparameter grid for Decision Tree
    decision_tree_params = {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]}

    # Hyperparameter grid for KNN
    knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    # Hyperparameter grid for Stacking
    stacking_params = {
        "final_estimator__fit_intercept": [True, False],
    }

    # Hyperparameter grid for Bagging
    bagging_params = {
        "n_estimators": [10, 20, 30],
        "estimator__max_depth": [None, 5, 10],
        "estimator__min_samples_split": [2, 5, 10],
    }

    # Hyperparameter grid for Boosting
    boosting_params = {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "estimator__max_depth": [None, 5],
        "estimator__min_samples_split": [10],
    }

    # Hyperparameter grid for Voting
    voting_params = {"weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1]]}

    # Define the parameter grid for each model
    param_grids = [
        linear_regression_params,
        decision_tree_params,
        knn_params,
        stacking_params,
        bagging_params,
        boosting_params,
        voting_params,
    ]

    # Perform hyperparameter tuning for each model
    best_estimators = []
    best_scores = []

    for (model_name, model), param_grid in zip(models, param_grids):
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(
            model, param_grid, cv=k, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_train_preprocessed, y_train_normalized.ravel())
        best_estimators.append((model_name, grid_search.best_estimator_))
        best_scores.append((-1) * grid_search.best_score_)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Mean MSE: {-1 * grid_search.best_score_}\n")

    # Evaluate the best models using k-fold cross-validation
    mse_scores = []
    for model_name, model in best_estimators:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(
            model,
            X_train_preprocessed,
            y_train_normalized.ravel(),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        mse_scores.append((-1) * scores)
        mean_mse = np.mean((-1) * scores)
        print(f"{model_name}: Mean MSE = {mean_mse}")

    # One of the best models which have feature importances are fitted for exploration
    # fitting the best model
    decision_tree = DecisionTreeRegressor().fit(
        X_train_preprocessed, y_train_normalized.ravel()
    )

    # Retrieve column names from ColumnTransformer
    categorical_transformer = preprocessor.named_transformers_["cat"]
    categorical_feature_names = categorical_transformer.named_steps[
        "onehot"
    ].get_feature_names_out(input_features=categorical_cols)
    feature_names = list(categorical_feature_names) + numerical_cols

    # Convert X_train_preprocessed to a DataFrame
    X_train_preprocessed_df = pd.DataFrame(
        X_train_preprocessed.toarray(), columns=feature_names
    )

    # Obtain feature importances
    if hasattr(decision_tree, "feature_importances_"):
        feature_importances = decision_tree.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]

        # Print feature importances
        print("Feature Importances for Decision Tree:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {feature_importances[i]}")
    else:
        print("The Decision Tree model does not have feature importances.")


def machine_leaning_best_model_quantity(df):
    # Remover outliers
    df = treat_outliers(df, "Quantity Sold (liters/kg)")
    # Preparando os dados para modelagem
    X = df.drop(
        columns=[
            "Approx. Total Revenue(INR)",
            "Price per Unit (sold)",
            "Quantity Sold (liters/kg)",
            "Quantity in Stock (liters/kg)",
        ]
    )
    y = df["Quantity Sold (liters/kg)"]

    # Dividindo em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = [
        "Location",
        "Farm Size",
        "Product Name",
        "Brand",
        "Customer Location",
        "Sales Channel",
    ]
    numerical_cols = [
        "Total Land Area (acres)",
        "Number of Cows",
        "Quantity (liters/kg)",
        "Price per Unit",
        "Total Value",
        "Shelf Life (days)",
        # "Quantity in Stock (liters/kg)",
        "Minimum Stock Threshold (liters/kg)",
        "Reorder Quantity (liters/kg)",
        "Days Before Expire",
        "Days to Sell",
    ]

    categorical_features = [
        "Location",
        "Product Name",
        "Brand",
        "Storage Condition",
        "Customer Location",
        "Sales Channel",
    ]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(), categorical_features)],
        remainder="passthrough",
    )

    cat_preprocessor = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_preprocessor = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_preprocessor, categorical_cols),
            ("num", num_preprocessor, numerical_cols),
        ]
    )

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    scaler = StandardScaler()
    y_train_normalized = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_normalized = scaler.transform(y_test.values.reshape(-1, 1))

    base_models = [
        ("linear_regression", LinearRegression()),
        ("decision_tree", DecisionTreeRegressor()),
        ("knn", KNeighborsRegressor()),
    ]

    # Define the stacking ensemble model
    stacking_model = StackingRegressor(
        estimators=base_models, final_estimator=LinearRegression()
    )

    # Define the bagging ensemble model
    bagging_model = BaggingRegressor(
        estimator=DecisionTreeRegressor()
    )  # Corrigido: usar 'estimator' em vez de 'base_estimator'

    # Definir o modelo de ensemble boosting
    boosting_model = AdaBoostRegressor(estimator=DecisionTreeRegressor())

    # Define the voting ensemble model
    voting_model = VotingRegressor(estimators=base_models)

    models = [
        ("Linear Regression", LinearRegression()),
        ("Decision Tree", DecisionTreeRegressor()),
        ("KNN", KNeighborsRegressor()),
        ("Stacking", stacking_model),
        ("Bagging", bagging_model),
        ("Boosting", boosting_model),
        ("Voting", voting_model),
    ]

    k = 5  # Number of folds
    mse_scores = []

    for model_name, model in models:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(
            model,
            X_train_preprocessed,
            y_train_normalized.ravel(),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        mse_scores.append((-1) * scores)
        mean_mse = np.mean((-1) * scores)

        print(f"{model_name} - Mean MSE: {mean_mse:.4f}")

    # Hyperparameter grid for Linear Regression
    linear_regression_params = {
        "fit_intercept": [True, False],
    }

    # Hyperparameter grid for Decision Tree
    decision_tree_params = {"max_depth": [None, 5, 10], "min_samples_split": [2, 5, 10]}

    # Hyperparameter grid for KNN
    knn_params = {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]}

    # Hyperparameter grid for Stacking
    stacking_params = {
        "final_estimator__fit_intercept": [True, False],
    }

    # Hyperparameter grid for Bagging
    bagging_params = {
        "n_estimators": [10, 20, 30],
        "estimator__max_depth": [None, 5, 10],
        "estimator__min_samples_split": [2, 5, 10],
    }

    # Hyperparameter grid for Boosting
    boosting_params = {
        "n_estimators": [50],
        "learning_rate": [0.1],
        "estimator__max_depth": [None, 5],
        "estimator__min_samples_split": [10],
    }

    # Hyperparameter grid for Voting
    voting_params = {"weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1]]}

    # Define the parameter grid for each model
    param_grids = [
        linear_regression_params,
        decision_tree_params,
        knn_params,
        stacking_params,
        bagging_params,
        boosting_params,
        voting_params,
    ]

    # Perform hyperparameter tuning for each model
    best_estimators = []
    best_scores = []

    for (model_name, model), param_grid in zip(models, param_grids):
        print(f"Tuning hyperparameters for {model_name}...")
        grid_search = GridSearchCV(
            model, param_grid, cv=k, scoring="neg_mean_squared_error", n_jobs=-1
        )
        grid_search.fit(X_train_preprocessed, y_train_normalized.ravel())
        best_estimators.append((model_name, grid_search.best_estimator_))
        best_scores.append((-1) * grid_search.best_score_)
        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best Mean MSE: {-1 * grid_search.best_score_}\n")

    # Evaluate the best models using k-fold cross-validation
    mse_scores = []
    for model_name, model in best_estimators:
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(
            model,
            X_train_preprocessed,
            y_train_normalized.ravel(),
            cv=cv,
            scoring="neg_mean_squared_error",
        )
        mse_scores.append((-1) * scores)
        mean_mse = np.mean((-1) * scores)
        print(f"{model_name}: Mean MSE = {mean_mse}")

    # One of the best models which have feature importances are fitted for exploration
    # fitting the best model
    decision_tree_quantity = DecisionTreeRegressor().fit(
        X_train_preprocessed, y_train_normalized.ravel()
    )

    # Retrieve column names from ColumnTransformer
    categorical_transformer = preprocessor.named_transformers_["cat"]
    categorical_feature_names = categorical_transformer.named_steps[
        "onehot"
    ].get_feature_names_out(input_features=categorical_cols)
    feature_names = list(categorical_feature_names) + numerical_cols

    # Convert X_train_preprocessed to a DataFrame
    X_train_preprocessed_df = pd.DataFrame(
        X_train_preprocessed.toarray(), columns=feature_names
    )

    # Obtain feature importances
    if hasattr(decision_tree_quantity, "feature_importances_"):
        feature_importances = decision_tree_quantity.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]

        # Print feature importances
        print("Feature Importances for Decision Tree:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {feature_importances[i]}")
    else:
        print("The Decision Tree model does not have feature importances.")

    return decision_tree_quantity


def predict(decision_tree_quantity):
    # Example input data (replace with your actual data)
    new_data = pd.DataFrame(
        {
            "Location": ["Delhi"],
            "Farm Size": ["Medium"],
            "Product Name": ["Milk"],
            "Brand": ["Amul"],
            "Customer Location": ["Delhi"],
            "Sales Channel": ["Retail"],
            "Total Land Area (acres)": [500],
            "Number of Cows": [50],
            "Quantity (liters/kg)": [1000],
            "Price per Unit": [50],
            "Total Value": [50000],
            "Shelf Life (days)": [10],
            "Quantity in Stock (liters/kg)": [200],
            "Minimum Stock Threshold (liters/kg)": [50],
            "Reorder Quantity (liters/kg)": [150],
            "Days Before Expire": [5],
            "Days to Sell": [3],
        }
    )

    # Preprocess the input data
    scaler = StandardScaler()

    cat_preprocessor = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_preprocessor = Pipeline([("scaler", StandardScaler())])

    categorical_cols = [
        "Location",
        "Farm Size",
        "Product Name",
        "Brand",
        "Customer Location",
        "Sales Channel",
    ]
    numerical_cols = [
        "Total Land Area (acres)",
        "Number of Cows",
        "Quantity (liters/kg)",
        "Price per Unit",
        "Total Value",
        "Shelf Life (days)",
        "Quantity in Stock (liters/kg)",
        "Minimum Stock Threshold (liters/kg)",
        "Reorder Quantity (liters/kg)",
        "Days Before Expire",
        "Days to Sell",
    ]

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_preprocessor, categorical_cols),
            ("num", num_preprocessor, numerical_cols),
        ]
    )

    new_data_preprocessed = preprocessor.transform(new_data)

    # Make predictions using the trained model
    predictions = decision_tree_quantity.predict(new_data_preprocessed)

    # If the target variable was normalized, reverse the normalization
    predictions_original_scale = scaler.inverse_transform(predictions.reshape(-1, 1))

    print("Predicted Quantity:", predictions_original_scale)
    return predictions_original_scale


dados_path = r"C:\Users\alexa\OneDrive\Anexos\Fiap\projeto_fase4\Enterprise Challenge\data_analysis\data\dados.csv"
df = pd.read_csv(dados_path)

df["Date"] = pd.to_datetime(df["Date"])
df["Daily_Sales_Rate"] = df.groupby("Product Name")[
    "Quantity Sold (liters/kg)"
].transform(lambda x: x.expanding().mean())

# 2. Dias até reabastecimento necessário
df["Days_to_Reorder"] = (
    df["Quantity (liters/kg)"] - df["Minimum Stock Threshold (liters/kg)"]
) / df["Daily_Sales_Rate"]

decision_tree_quantity = machine_leaning_best_model_quantity(df)

predict(decision_tree_quantity)
