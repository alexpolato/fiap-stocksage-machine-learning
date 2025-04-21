import pandas as pd
import numpy as np
import pickle
import datetime

# Carregar modelo e pré-processadores
final_model = pd.read_pickle("dairy/dairy_models/final_model.pkl")
with open(
    "dairy/dairy_models/preprocessor.pkl", "rb"
) as f:  # Substitua pelo seu pré-processador
    preprocessor = pickle.load(f)

with open("dairy/dairy_data/feature_lists.pkl", "rb") as f:
    feature_lists = pickle.load(f)

# feature_lists = pd.DataFrame(feature_lists)
all_features = (
    feature_lists["categorical_features"] + feature_lists["numerical_features"]
)
print("All features pré processando \n\n", all_features, "\n\n")

# Definir variáveis para cada coluna
current_date = datetime.datetime.now()
production_date = current_date - datetime.timedelta(days=10)
expiration_date = production_date + datetime.timedelta(days=65)

product_name = "Ice Cream"
atual_quantity = 800
price_per_unit = 28.0
shelf_life = (expiration_date - production_date).days
quantity_sold = 500
price_per_unit_sold = 29.0
days_to_expire = (expiration_date - current_date).days
days_to_sell = (current_date - production_date).days
quantity_in_stock = atual_quantity - quantity_sold
sales_velocity = quantity_sold / days_to_sell
quantity_able_to_sell_before_expire = sales_velocity * days_to_expire
# Correct conditional logic for quantity_lost
if quantity_able_to_sell_before_expire >= quantity_in_stock:
    quantity_lost = 0
else:
    quantity_lost = quantity_in_stock - quantity_able_to_sell_before_expire

stock_efficiency = (sales_velocity * shelf_life) / atual_quantity
value_lost = quantity_lost * price_per_unit
revenue_before_losses = quantity_sold * (price_per_unit_sold - price_per_unit)

# Calcular valores derivados
total_value = atual_quantity * price_per_unit

# Criar dicionário de dados de exemplo
example_data = {
    "Product Name": [product_name],
    "Quantity (liters/kg)": [atual_quantity],
    "Price per Unit": [price_per_unit],
    "Total Value": [total_value],
    "Shelf Life (days)": [shelf_life],
    "Quantity Sold (liters/kg)": [quantity_sold],
    "Price per Unit (sold)": [price_per_unit_sold],
    "Quantity in Stock (liters/kg)": [quantity_in_stock],
    "Days_to_Sell": [days_to_sell],
    "Days_to_Expire": [days_to_expire],
    "Sales_Velocity": [sales_velocity],
    "Quantity_Abble_Sell_Before_Expire": [quantity_able_to_sell_before_expire],
    "Quantity_Lost": [quantity_lost],
    "Stock_Efficiency": [stock_efficiency],
    "Value_Lost": [value_lost],
    "Revenue_Before_Losses": [revenue_before_losses],
}

# Converter para DataFrame
example_df = pd.DataFrame(example_data)
print("Exemplo de dados para teste de predição:\n", example_df)


X_novo = pd.DataFrame(example_df)

# Converter para DataFrame
# Aplicar pré-processamento
X_processed = preprocessor.transform(X_novo)

# Fazer predição
prediction = final_model.predict(X_processed)
print("Predição:", prediction)
