import os
import numpy as np
from keras.models import load_model
from PIL import Image

# Configuracion
MODELO_PATH = 'mejor_modelo_animales.h5' 
RUTA_CARPETA_TEST = 'D:\\Inteligencia_artificial\\Proyecto2\\test' 
IMG_SIZE = (64, 64) 


CLASES = ['Hormiga', 'Perro', 'Gato', 'Tortuga', 'Mariquita'] 


print(f"Cargando modelo desde: {MODELO_PATH}...")
try:
    model = load_model(MODELO_PATH)
    print(" Modelo cargado correctamente.\n")
except Exception as e:
    print(f" Error al cargar modelo: {e}")
    exit()


print(f"Analizando imágenes en: {RUTA_CARPETA_TEST}")
print("-" * 60)
print(f"{'NOMBRE DEL ARCHIVO':<35} | {'PREDICCIÓN':<15} | {'SEGURIDAD'}")
print("-" * 60)

archivos = os.listdir(RUTA_CARPETA_TEST)

for archivo in archivos:
    if archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        ruta_completa = os.path.join(RUTA_CARPETA_TEST, archivo)
        
        try:
            # Cargar imagen
            img = Image.open(ruta_completa)
            img = img.convert('RGB')
            img = img.resize(IMG_SIZE)
            
            # Convertir a números
            img_array = np.array(img) 
            img_array = img_array.astype('float32') 
            
            # Formato de batch (1, 64, 64, 3)
            img_procesada = np.expand_dims(img_array, axis=0)
            
            #Predicción
            prediccion = model.predict(img_procesada, verbose=0)
            
            indice_ganador = np.argmax(prediccion)
            nombre_clase = CLASES[indice_ganador]
            probabilidad = prediccion[0][indice_ganador] * 100
            
            print(f"{archivo:<35} | {nombre_clase:<15} | {probabilidad:.1f}%")
            
        except Exception as e:
            print(f" Error con {archivo}: {e}")

print("-" * 60)