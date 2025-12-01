import pandas as pd

# 1. Cargar datos originales
archivo_entrada = 'dataset_sintetico_5000_ampliado.csv'
df = pd.read_csv(archivo_entrada)

print(f"Filas iniciales: {len(df)}")

# 2. Limpieza (Eliminar textos repetidos)
# subset=['texto']: Solo mira la columna de texto para decidir si es duplicado
df_limpio = df.drop_duplicates(subset=['texto'], keep='first').copy()

# Resetear el índice (para que los números de fila vuelvan a ser 0, 1, 2...)
df_limpio.reset_index(drop=True, inplace=True)

print(f"Filas finales tras limpieza: {len(df_limpio)}")
print(f"Se eliminaron {len(df) - len(df_limpio)} registros duplicados.")

# 3. GUARDAR EL ARCHIVO LIMPIO
archivo_salida = 'dataset_limpio.csv'

# index=False es CLAVE: evita que Pandas te agregue una columna extra de números (0,1,2...) al inicio
df_limpio.to_csv(archivo_salida, index=False, encoding='utf-8')


# Muestra una única fila de ejemplo por cada texto que se repite en el dataset
duplicados_unicos = df[df.duplicated(subset=['texto'], keep=False)].drop_duplicates(subset=['texto'])

# Visualizar solo las columnas clave para facilitar la lectura
print(duplicados_unicos[['id', 'texto']].head(10))


print(f"\n¡Listo! Archivo creado exitosamente: {archivo_salida}")