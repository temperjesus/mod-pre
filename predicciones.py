import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

# Cargar datos
file_path = "predicciones.xlsx"
df = pd.read_excel(file_path)

# Limpieza de nombres de columnas
df.columns = df.columns.str.strip()

# Verificar que las columnas requeridas existan
required_columns = {'fecha', 'precio_kwh'}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    print(f"Error: Faltan las siguientes columnas en el archivo: {missing_columns}")
    exit()

# Convertir fechas a formato datetime
df['fecha'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')

# Convertir fechas a números ordinales
df['fecha_ordinal'] = df['fecha'].map(datetime.toordinal)

# Definir IPC estimado según los años futuros
ipc_futuro = {
    2026: 5.23,
    2027: 6.43,
    2028: 4.57,
    2029: 8.40
}

# Calcular IPC promedio si el año no está en el diccionario
ipc_promedio = np.mean(list(ipc_futuro.values()))

# Entrenar modelo Random Forest
X = df[['fecha_ordinal']]
y = df['precio_kwh']

modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Función para calcular inflación acumulada
def calcular_inflacion_acumulada(anio_inicio, anio_fin):
    """Calcula el factor de inflación acumulado desde anio_inicio hasta anio_fin."""
    factor = 1.0
    for anio in range(anio_inicio, anio_fin + 1):
        ipc = ipc_futuro.get(anio, ipc_promedio)  # Si no hay dato, usa IPC promedio
        factor *= (1 + ipc / 100)
    return factor

# Función para predecir el precio con inflación acumulada
def predecir_precio(fecha_str):
    fecha_pred = datetime.strptime(fecha_str, '%Y-%m-%d')
    fecha_ordinal_pred = fecha_pred.toordinal()
    
    precio_base = modelo.predict([[fecha_ordinal_pred]])[0]
    
    # Ajustar por inflación acumulada desde el último año de datos hasta la fecha predicha
    anio_inicio = df['fecha'].dt.year.max()  # Último año con datos reales
    anio_pred = fecha_pred.year
    factor_inflacion = calcular_inflacion_acumulada(anio_inicio, anio_pred)
    
    precio_ajustado = precio_base * factor_inflacion

    return precio_ajustado, precio_base, factor_inflacion

# Pedir fecha al usuario
fecha_usuario = input("Ingrese la fecha (YYYY-MM-DD) para predecir el precio: ")
precio_ajustado, precio_base, factor_inflacion = predecir_precio(fecha_usuario)

print(f"Precio estimado sin inflación: {precio_base:.4f}")
print(f"Factor de inflación acumulado: {factor_inflacion:.4f}")
print(f"Precio ajustado con inflación acumulada: {precio_ajustado:.4f}")

# Graficar resultados
plt.scatter(df['fecha'], df['precio_kwh'], color='blue', label='Datos Reales')
plt.scatter(datetime.strptime(fecha_usuario, '%Y-%m-%d'), precio_ajustado, color='red', label='Predicción Ajustada')
plt.xlabel('Fecha')
plt.ylabel('Precio kWh')
plt.legend()
plt.xticks(rotation=45)
plt.show()
