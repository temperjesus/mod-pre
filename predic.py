import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Leer el archivo Excel
data = pd.read_excel('datos_kwh_costa_caribe.xlsx')  # o usa .read_csv si es CSV
print(f"Número de filas en el DataFrame después de la lectura: {len(data)}")
print(data.head())

# 2. Convertir la columna 'Fecha' a datetime si es necesario
if 'Fecha' in data.columns:
    data['Fecha'] = pd.to_datetime(data['Fecha'])

# 3. Imputar valores NaN con la media (antes de crear los lags)
data = data.fillna(data.mean())
print(f"Número de filas después de imputar NaN: {len(data)}")

# 4. Crear variables lagged (retardadas) para 'Precio_kWh'
for i in range(1, 13):
    data[f'Precio_kWh_lag{i}'] = data['Precio_kWh'].shift(i)

# 5. Eliminar filas con valores NaN después de crear los lags
data = data.dropna()
print(f"Número de filas después de crear los lags y eliminar NaN: {len(data)}")

# 6. Preparar los datos para el modelo
# Eliminar la columna 'Fecha' si está presente
if 'Fecha' in data.columns:
    data = data.drop('Fecha', axis=1)

# X: variables predictoras
X = data.drop('Precio_kWh', axis=1)
# y: variable objetivo
y = data['Precio_kWh']

# 7. Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Inicializar y entrenar el modelo Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# 10. Evaluar el modelo
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f'MAE: {mae}')
print(f'MSE: {mse}')

# 11. Visualizar las predicciones vs los valores reales
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Valores Reales', marker='o')
plt.plot(y_test.index, y_pred, label='Predicciones', marker='x')
plt.xlabel('Índice de la muestra (mes)')
plt.ylabel('Precio del kWh')
plt.title('Predicciones del Precio del kWh vs Valores Reales')
plt.legend()
plt.grid(True)
plt.show()

# 12. Predicción para el futuro (ejemplo: un mes adelante)
# Supongamos que tenemos los valores para el mes siguiente
new_data = pd.DataFrame({
    'Precip': [160.0],      # Ejemplo
    'Temp': [28.5],        # Ejemplo
    'Gen_Solar': [470.0],   # Ejemplo
    'Gen_Eolica': [315.0],  # Ejemplo
    'Demanda': [1190.0]     # Ejemplo
})

# Crear los lags para el nuevo dato usando los últimos 12 meses *reales*
for i in range(1, 13):
    new_data[f'Precio_kWh_lag{i}'] = data['Precio_kWh'].iloc[-i]

# Realizar la predicción
predicted_price = model.predict(new_data)
print(f"Predicción del precio del kWh para el próximo mes: {predicted_price[0]}")
