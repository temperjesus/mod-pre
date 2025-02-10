import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar los datos desde el archivo Excel
df = pd.read_excel("datos_kwh_costa_caribe.xlsx")

# Asegurar que la columna Fecha sea de tipo datetime
df["Fecha"] = pd.to_datetime(df["Fecha"])

# Ordenar por fecha
df = df.sort_values(by="Fecha")

# Codificar el estrato social
label_encoder = LabelEncoder()
df["Estrato_Cod"] = label_encoder.fit_transform(df["Estrato"])

# Variables de entrada (X) y salida (y)
X = df[["Fecha", "Estrato_Cod", "IPC"]]
y = df["Precio_KWh"]

# Convertir fechas a números para el modelo
X["Fecha"] = X["Fecha"].map(pd.Timestamp.toordinal)

# Separar datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
modelo = RandomForestRegressor(n_estimators=200, random_state=42)
modelo.fit(X_train, y_train)

# Generar predicciones para los próximos 20 años
futuras_fechas = pd.date_range(start=df["Fecha"].max() + pd.DateOffset(months=1), periods=240, freq="M")
futuras_fechas_ordinal = futuras_fechas.map(pd.Timestamp.toordinal)

# Simular estratos y una inflación creciente del 7% anual
estratos_pred = [1, 2, 3, 4, 5, 6] * 40  # Diversificar estratos en la predicción
ipc_inicial = df["IPC"].iloc[-1]
ipc_pred = [ipc_inicial * (1.07 ** (i / 12)) for i in range(240)]

# Crear DataFrame con datos de predicción
df_pred = pd.DataFrame({
    "Fecha": futuras_fechas,
    "Estrato_Cod": estratos_pred[:240],
    "IPC": ipc_pred
})

# Convertir fechas a ordinales
df_pred["Fecha"] = df_pred["Fecha"].map(pd.Timestamp.toordinal)

# Hacer predicciones
predicciones = modelo.predict(df_pred)
df_pred["Precio_KWh"] = predicciones

# Guardar las predicciones en Excel
output_file = "predicciones_kwh.xlsx"
df_pred.to_excel(output_file, index=False)

# Graficar las predicciones
plt.figure(figsize=(10, 5))
plt.plot(df_pred["Fecha"], df_pred["Precio_KWh"], marker="o", linestyle="-", label="Predicción")
plt.xlabel("Fecha")
plt.ylabel("Precio KWh")
plt.title("Predicción del Precio del KWh en la Costa Caribe")
plt.legend()
plt.xticks(rotation=45)
plt.grid()
plt.show()

print(f"Predicciones guardadas en: {output_file}")
