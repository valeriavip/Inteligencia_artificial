import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Stopwords en español
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
lista_stopwords = stopwords.words('spanish')

archivo = 'dataset_sintetico_5000_ampliado.csv'
df = pd.read_csv(archivo)

textos = df['texto'].fillna('').tolist()

# --- Muestreo ---
# Para visualizar un heatmap 
cantidad_muestra = 1000
textos_muestra = textos[:cantidad_muestra]
ids_muestra = df['id'][:cantidad_muestra].astype(str).tolist()

# --- Bag of Words ---
count_vectorizer = CountVectorizer(stop_words=lista_stopwords)
matrix = count_vectorizer.fit_transform(textos_muestra)

# Convertir a DataFrame para ver las frecuencias 
data_matrix = matrix.todense()
frecuencias = pd.DataFrame(data_matrix, 
                           columns=count_vectorizer.get_feature_names_out(), 
                           index=ids_muestra)

# Palabras más comunes en el primer tweet de la muestra
print("Frecuencia de palabras en el primer tweet:")
print(frecuencias.iloc[0].sort_values(ascending=False).head(5))

# --- Similitud del Coseno ---
salida = cosine_similarity(matrix)

# --- Visualización ---
plt.figure(figsize=(12, 10))
plt.imshow(salida, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Grado de Similitud (0 a 1)')

# Configurar ejes con los IDs
plt.xticks(range(len(ids_muestra)), ids_muestra, rotation=90)
plt.yticks(range(len(ids_muestra)), ids_muestra)

plt.title(f'Matriz de Similitud del Coseno (Muestra de {cantidad_muestra} tweets)')
plt.xlabel('ID del Tweet')
plt.ylabel('ID del Tweet')

plt.tight_layout()
plt.show()
