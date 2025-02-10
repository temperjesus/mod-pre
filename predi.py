import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde el archivo Excel
df = pd.read_excel("datos_kwh_2015_2024_mejorado.xlsx")

# Asegurar que los nombres de las columnas sean correctos
print("Columnas del archivo:", df.columns)

# Convertir fecha a formato numérico para el modelo
df["Fecha"] = pd.to_datetime(df["Fecha"])
df["Fecha_Num"] = df["Fecha"].map(pd.Timestamp.toordinal)

# Variables predictoras y variable objetivo
X = df[["Fecha_Num", "IPC", "Estrato"]]
y = df["Precio_KWh"]

# Dividir en datos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Generar predicciones futuras desde la última fecha registrada
dias_futuros = 10 * 12  # 10 años en meses
ultima_fecha = df["Fecha"].max()
fechas_pred = [ultima_fecha + pd.DateOffset(months=i) for i in range(1, dias_futuros + 1)]

# Convertir fechas futuras a números
fechas_pred_num = [fecha.toordinal() for fecha in fechas_pred]

# Simular IPC e estrato (aquí puedes usar tendencias reales)
ipc_pred = np.linspace(df["IPC"].mean(), df["IPC"].mean() * 1.5, dias_futuros)
estrato_pred = np.random.choice(df["Estrato"].unique(), dias_futuros)

# Crear DataFrame de predicción
df_pred = pd.DataFrame({
    "Fecha": fechas_pred,
    "Fecha_Num": fechas_pred_num,
    "IPC": ipc_pred,
    "Estrato": estrato_pred
})

# Predecir precios del kWh
df_pred["Precio_KWh"] = modelo.predict(df_pred[["Fecha_Num", "IPC", "Estrato"]])

# Guardar predicciones en un nuevo archivo Excel
df_pred.to_excel("predicciones_kwh.xlsx", index=False)

# Graficar la predicción
plt.figure(figsize=(12, 6))
plt.plot(df["Fecha"], df["Precio_KWh"], label="Histórico", linestyle="--", marker="o")
plt.plot(df_pred["Fecha"], df_pred["Precio_KWh"], label="Predicción", linestyle="-", marker="o")
plt.xlabel("Fecha")
plt.ylabel("Precio KWh")
plt.title("Predicción del Precio del KWh en la Costa Caribe")
plt.legend()
plt.grid()
plt.xticks(rotation=45)
plt.show()
