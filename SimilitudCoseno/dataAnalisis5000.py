import pandas as pd

# Cargar dataset
archivo_entrada = 'dataset_sintetico_5000_ampliado.csv'
df = pd.read_csv(archivo_entrada)

print(f"Filas iniciales: {len(df)}")

df_limpio = df.drop_duplicates(subset=['texto'], keep='first').copy()

# Reiniciar índices
df_limpio.reset_index(drop=True, inplace=True)

print(f"Filas limpias: {len(df_limpio)}")
print(f"{len(df) - len(df_limpio)} registros duplicados.")

archivo_salida = 'dataset_limpio.csv'


df_limpio.to_csv(archivo_salida, index=False, encoding='utf-8')


duplicados_unicos = df[df.duplicated(subset=['texto'], keep=False)].drop_duplicates(subset=['texto'])

print(duplicados_unicos[['id', 'texto']].head(10))


print(f"\n Archivo creado exitosamente: {archivo_salida}")