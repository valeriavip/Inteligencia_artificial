import os
import re
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, LeakyReLU
import keras
from keras.layers import Rescaling, BatchNormalization, GlobalAveragePooling2D
from PIL import Image
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Configuracion
dirname = os.path.join('D:\\DatasetProyecto')

imgpath = dirname + os.sep
ladybug = "Mariquita" 

images = []
directories = []
dircount = []
prevRoot = ''
cant = 0

IMG_SIZE = (64, 64)

print(f"Buscando imágenes en: {dirname}")

for root, dirnames, filenames in os.walk(imgpath):
    nombre_actual = os.path.basename(root)
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
            filepath = os.path.join(root, filename)
            
            try:
                imagen_original = Image.open(filepath)
                imagen_original = imagen_original.resize(IMG_SIZE)
                imagen_original = imagen_original.convert('RGB')


                datos_original = np.array(imagen_original)
                images.append(datos_original)
                cant = cant + 1 
                
                #nombre carpeta
                nombre_actual_carpeta = os.path.basename(root)
                if nombre_actual_carpeta != ladybug:
                    img_espejo = imagen_original.transpose(Image.FLIP_LEFT_RIGHT)
                    datos_espejo = np.array(img_espejo)
                    images.append(datos_espejo)
                    cant = cant + 1

                nombre_actual = os.path.basename(root)
                msg = f"Carpeta: '{nombre_actual}' / Imágenes: {cant}"
                print(msg + " " * 10, end="\r")
            except Exception as e:
                print(f"Error con archivo {filename}: {e}")

#Contar imagenes
    if prevRoot != root:
        if cant > 0: 
            print(root, cant)
            directories.append(root)
            dircount.append(cant)
            cant = 0
        prevRoot = root

if cant > 0:
    dircount.append(cant)

print('\nCarpetas leídas con imágenes:', len(directories))
print("Imágenes por clase:", dircount)
print('Total de imágenes:', sum(dircount))

# Etiquetas
labels = []
indice = 0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice = indice + 1

print("\nEtiquetas asignadas:")
deportes = []
indice = 0
for directorio in directories:
    name = directorio.split(os.sep)[-1]
    print(f"Indice: {indice} -> Clase: {name}")
    deportes.append(name)
    indice = indice + 1

y = np.array(labels)
X = np.array(images, dtype=np.uint8)

classes = np.unique(y)
nClasses = len(classes)
print('Total de clases: ', nClasses)



train_X, test_X, train_Y, test_Y = train_test_split(X, y, test_size=0.2, random_state=42)

# Etiqueta -> Vector 
train_Y_one_hot = to_categorical(train_Y, num_classes=nClasses)
test_Y_one_hot = to_categorical(test_Y, num_classes=nClasses)

# Sub-set de validación
train_X, valid_X, train_label, valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13)

print('\nTamaños finales:')
print('Entrenamiento:', train_X.shape)
print('Validación:', valid_X.shape)

# Modelo CNN
INIT_LR = 1e-3
epochs = 50 
batch_size = 64

animal_model = Sequential()

animal_model.add(Rescaling(1./255, input_shape=(IMG_SIZE[1], IMG_SIZE[0], 3)))

# BLOQUE 1 
animal_model.add(Conv2D(32, (3, 3), padding='same', use_bias=False))
animal_model.add(BatchNormalization())
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2)))
animal_model.add(Dropout(0.3)) 

# BLOQUE 2 
animal_model.add(Conv2D(64, (3, 3), padding='same', use_bias=False))
animal_model.add(BatchNormalization())
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2)))
animal_model.add(Dropout(0.4)) 

# BLOQUE 3 
animal_model.add(Conv2D(128, (3, 3), padding='same', use_bias=False))
animal_model.add(BatchNormalization())
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(MaxPooling2D((2, 2)))
animal_model.add(Dropout(0.5)) 

# SALIDA 
animal_model.add(GlobalAveragePooling2D())
animal_model.add(Dense(128, activation='linear'))
animal_model.add(LeakyReLU(alpha=0.1))
animal_model.add(Dropout(0.5))
animal_model.add(Dense(nClasses, activation='softmax'))

animal_model.summary()

animal_model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adam(learning_rate=INIT_LR),
    metrics=['accuracy']
)

# Mejor modelo y Early Stopping
checkpoint = ModelCheckpoint(
    'mejor_modelo_animales.h5', 
    monitor='val_accuracy',     
    save_best_only=True,        
    mode='max',                
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,                 
    restore_best_weights=True   
)

print("\n--- INICIANDO ENTRENAMIENTO ---")
history = animal_model.fit(
    train_X,
    train_label,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(valid_X, valid_label),
    callbacks=[checkpoint, early_stopping]
)
print("\n Entrenamiento finalizado. El mejor modelo se guardó como 'mejor_modelo_animales.h5'")