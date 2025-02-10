import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Cargar los datos desde el archivo Excel
df = pd.read_excel("datos_kwh_ajustados.xlsx")

# Definir las variables de entrada y salida
X = df[["Año", "Mes", "Precip", "Temp", "Gen_Solar", "Gen_Eolica", "Demanda", "IPC", "Variación_IPC", "Estrato"]]
y = df["Precio_kWh"]

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

# Función para predecir el precio del kWh en una fecha dada y estrato específico
def predecir_precio(fecha_prediccion, estrato):
    año_pred = fecha_prediccion.year
    mes_pred = fecha_prediccion.month
    
    # Valores promedio para otras variables
    precip_pred = df["Precip"].mean()
    temp_pred = df["Temp"].mean()
    solar_pred = df["Gen_Solar"].mean()
    eolica_pred = df["Gen_Eolica"].mean()
    demanda_pred = df["Demanda"].mean()
    ipc_pred = df["IPC"].iloc[-1] * 1.05  # Suposición de inflación anual del 5%
    variacion_ipc_pred = 0.05
    
    estrato = max(1, min(estrato, 6))  # Asegurar que el estrato esté entre 1 y 6
    
    X_pred = [[año_pred, mes_pred, precip_pred, temp_pred, solar_pred, eolica_pred, demanda_pred, ipc_pred, variacion_ipc_pred, estrato]]
    return modelo_rf.predict(X_pred)[0]

# Generar predicciones para los próximos 10 años (estrato 3 por defecto)
ultima_fecha = pd.to_datetime(df["Fecha"].max())
fechas_futuras = pd.date_range(start=ultima_fecha, periods=121, freq="MS")  # 10 años
precios_futuros = [predecir_precio(f, 3) for f in fechas_futuras]

# Graficar la predicción
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Precio_kWh"], label="Datos históricos", color="blue")
plt.plot(fechas_futuras, precios_futuros, label="Predicción (Estrato 3)", linestyle="dashed", color="red")
plt.xlabel("Fecha")
plt.ylabel("Precio del kWh (COP)")
plt.title("Predicción del Precio del kWh en la Costa Caribe Colombiana")
plt.legend()
plt.grid(True)
plt.show()
