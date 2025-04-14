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


def training_total_revenue(df):
    # Tratamento de outliers
    df = treat_outliers(df, "Approx. Total Revenue(INR)")

    # Preparando os dados para modelagem
    X = df.drop(
        columns=[
            "Approx. Total Revenue(INR)",
            "Price per Unit (sold)",
            "Quantity Sold (liters/kg)",
            "Quantity in Stock (liters/kg)",
            "Total Value",
        ]
    )
    y = df["Approx. Total Revenue(INR)"]

    # Dividindo em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Definindo colunas categóricas e numéricas
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
        # "Total Value",
        "Days_to_Reorder" "Shelf Life (days)",
        "Minimum Stock Threshold (liters/kg)",
        "Reorder Quantity (liters/kg)",
        "Days Before Expire",
        "Days to Sell",
    ]

    # Pré-processamento
    cat_preprocessor = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    num_preprocessor = Pipeline([("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        [
            ("cat", cat_preprocessor, categorical_cols),
            ("num", num_preprocessor, numerical_cols),
        ]
    )

    # Pré-processando os dados
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Normalizando o target
    scaler = StandardScaler()
    y_train_normalized = scaler.fit_transform(y_train.values.reshape(-1, 1))
    y_test_normalized = scaler.transform(y_test.values.reshape(-1, 1))

    # Definindo e ajustando o modelo Decision Tree
    decision_tree = DecisionTreeRegressor(random_state=42)

    # Otimização de hiperparâmetros
    decision_tree_params = {
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    print("Tuning hyperparameters for Decision Tree...")
    grid_search = GridSearchCV(
        decision_tree,
        decision_tree_params,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train_preprocessed, y_train_normalized.ravel())

    best_decision_tree = grid_search.best_estimator_
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Mean MSE: {-1 * grid_search.best_score_:.4f}\n")

    # Avaliação final com cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        best_decision_tree,
        X_train_preprocessed,
        y_train_normalized.ravel(),
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    mean_mse = np.mean((-1) * scores)
    print(f"Decision Tree - Final Mean MSE: {mean_mse:.4f}")

    # Análise de importância de features
    categorical_transformer = preprocessor.named_transformers_["cat"]
    categorical_feature_names = categorical_transformer.named_steps[
        "onehot"
    ].get_feature_names_out(input_features=categorical_cols)
    feature_names = list(categorical_feature_names) + numerical_cols

    if hasattr(best_decision_tree, "feature_importances_"):
        feature_importances = best_decision_tree.feature_importances_
        sorted_indices = np.argsort(feature_importances)[::-1]

        print("\nFeature Importances for Decision Tree:")
        for i in sorted_indices:
            print(f"{feature_names[i]}: {feature_importances[i]:.4f}")
    else:
        print("The Decision Tree model does not have feature importances.")

    # Retornando os objetos necessários
    total_rev = {
        "model": best_decision_tree,
        "preprocessor": preprocessor,
        "scaler": scaler,
        "feature_names": feature_names,
    }
    return total_rev


dados_path = r"C:\Users\alexa\OneDrive\Anexos\Fiap\projeto_fase4\Enterprise Challenge\data_analysis\data\dados.csv"
df = pd.read_csv(dados_path)
df["Daily_Sales_Rate"] = df.groupby("Product Name")[
    "Quantity Sold (liters/kg)"
].transform(lambda x: x.expanding().mean())

# 2. Dias até reabastecimento necessário
df["Days_to_Reorder"] = (
    df["Quantity (liters/kg)"] - df["Minimum Stock Threshold (liters/kg)"]
) / df["Daily_Sales_Rate"]

total_rev = training_total_revenue(df)
