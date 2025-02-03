import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# Cargar datos desde un archivo Excel
archivo = "predicciones.xlsx"
df = pd.read_excel(archivo)

# Normalizar nombres de columnas
df.columns = df.columns.str.strip().str.lower()

# Verificar si la columna "precio_kwh" existe
if "precio_kwh" not in df.columns:
    raise KeyError("La columna 'precio_kwh' no se encuentra en el archivo. Verifica el nombre de la columna en el Excel.")

# Convertir la columna fecha a tipo datetime
df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True, errors='coerce')

# Verificar si hay fechas nulas
if df["fecha"].isnull().any():
    raise ValueError("Se encontraron valores no válidos en la columna 'fecha'. Verifica el formato en el Excel.")

# Extraer año y mes como variables
fecha_inicial = df["fecha"].min()
df["anio_mes"] = df["fecha"].dt.to_period("M").astype(str)
df = df.groupby("anio_mes")["precio_kwh"].mean().reset_index()
df["anio_mes"] = pd.to_datetime(df["anio_mes"])
df["meses"] = (df["anio_mes"] - fecha_inicial).dt.days // 30  # Convertir fechas a meses desde inicio

# Variables de entrenamiento
X = df[["meses"]]
y = df["precio_kwh"]

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Crear y entrenar el modelo Random Forest
modelo = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
modelo.fit(X_train_scaled, y_train)

# Predicción para los próximos meses hasta el rango deseado (hasta 10 años = 120 meses)
rango_meses = 120  # Cambiar según la necesidad
dias_futuros = np.arange(df["meses"].max() + 1, df["meses"].max() + 1 + rango_meses).reshape(-1, 1)
dias_futuros_scaled = scaler.transform(dias_futuros)
predicciones = modelo.predict(dias_futuros_scaled)

# Crear un DataFrame con las predicciones
fechas_predichas = [fecha_inicial + pd.DateOffset(months=int(i)) for i in dias_futuros.flatten()]
df_predicciones = pd.DataFrame({"fecha": fechas_predichas, "precio_kwh": predicciones})

# Graficar resultados
plt.figure(figsize=(12, 6))
plt.scatter(df["anio_mes"], df["precio_kwh"], label="Datos reales", color='blue', alpha=0.6)
plt.plot(df_predicciones["fecha"], df_predicciones["precio_kwh"], label="Predicción", color='red', linewidth=2, marker='o')
plt.xlabel("Fecha")
plt.ylabel("Precio kWh")
plt.title("Predicción del precio del kWh en Barranquilla por meses")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(rotation=45)
plt.show()
