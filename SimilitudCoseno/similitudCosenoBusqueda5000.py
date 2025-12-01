
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

# Rellenar vacíos para evitar errores
df['texto'] = df['texto'].fillna('')

# --- Crear el "Índice" de Textos ---
# Usamos todo el dataset
print("Indexando dataset...")
count_vectorizer = CountVectorizer(stop_words=lista_stopwords)
matrix_dataset = count_vectorizer.fit_transform(df['texto']) # Aprende el vocabulario

# --- Motor de Búsqueda ---
def buscar_texto(consulta, top_n=100):
    """
    Busca la consulta en el dataset usando similitud del coseno.
    """
    query_vec = count_vectorizer.transform([consulta])
    
    # Calcular similitud coseno
    similitudes = cosine_similarity(query_vec, matrix_dataset).flatten()
    
    # Top N índices con mayor similitud
    indices_top = similitudes.argsort()[-top_n:][::-1]
    
    # Resultados
    resultados = []
    scores = []
    
    print(f"\n--- Resultados para: '{consulta}' ---")
    for i in indices_top:
        score = similitudes[i]
        if score > 0: # Mostramos si hay coincidencia
            id_tweet = df.iloc[i]['id']
            texto = df.iloc[i]['texto']
            resultados.append({'id': id_tweet, 'texto': texto, 'score': score})
            scores.append(score)
            print(f"[Similitud: {score:.4f}] ID {id_tweet}: {texto[:100]}...") # Muestra primeros 100 caracteres
    
    return pd.DataFrame(resultados), scores

# --- Búsqueda ---
mi_consulta = "Antes los proyectos de vida eran a largo plazo; hoy parece que nuestra identidad cambia tan rápido como las tendencias en TikTok." 

df_resultados, scores = buscar_texto(mi_consulta)

# --- Visualización ---
if not df_resultados.empty:
    plt.figure(figsize=(10, 5))
    # Gráfico
    plt.barh(df_resultados['id'].astype(str), df_resultados['score'], color='skyblue')
    plt.xlabel('Similitud del Coseno (0-1)')
    plt.ylabel('ID del Tweet')
    plt.title(f'Top coincidencias para: "{mi_consulta}"')
    plt.gca().invert_yaxis() # Invertir eje Y para que el #1 quede arriba
    plt.show()
else:
    print("No se encontraron coincidencias.")