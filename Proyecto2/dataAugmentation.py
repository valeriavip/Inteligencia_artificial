import os
from PIL import Image

# Configuración
ruta_carpeta_ladybug = "D:\\DatasetProyecto\\Mariquita" 

def generar_rotaciones(ruta):
    archivos = [f for f in os.listdir(ruta) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    print(f"Procesando {len(archivos)} imágenes en {ruta}...")

    for archivo in archivos:
        ruta_completa = os.path.join(ruta, archivo)
        try:
            img = Image.open(ruta_completa)
            
            # Generar rotaciones
            for angulo in [90, 180, 270]:
                img_rotada = img.rotate(angulo, expand=True)
                
                # Creamos un nombre nuevo
                nombre_base, ext = os.path.splitext(archivo)
                nuevo_nombre = f"{nombre_base}_rot{angulo}{ext}"
                ruta_guardado = os.path.join(ruta, nuevo_nombre)
                
                # Guardamos
                img_rotada.save(ruta_guardado)
                
        except Exception as e:
            print(f"Error en {archivo}: {e}")

    print("Generación de rotaciones completada")

# Ejecutar
generar_rotaciones(ruta_carpeta_ladybug)