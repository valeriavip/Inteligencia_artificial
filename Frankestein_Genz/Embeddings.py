import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Modelo pre-entrenado
model = SentenceTransformer("all-MiniLM-L6-v2")

# Base de datos de textos
#corpus = [
 #   "La inteligencia artificial es fascinante",
  #  "El aprendizaje automático mejora cada día",
   # "Python es un lenguaje poderoso",
#]


df = pd.read_csv("C:\\Users\\villi\\OneDrive\\Escritorio\\WebScrapping\\datasetTexto.csv")       # Carga el CSV
corpus = df["texto"].dropna().tolist()   # Convierte la columna a lista


# Consultas
query = "¿Qué es Frankestein?"

# Generar embeddings
corpus_embeddings = model.encode(corpus)
query_embedding = model.encode([query])

# Similitud coseno
similarities = cosine_similarity([query_embedding[0]], corpus_embeddings)[0]

# Resultados ordenados
sorted_idx = np.argsort(similarities)[::-1]
print("Resultados más relevantes:")
for idx in sorted_idx:
    print(f"Texto: {corpus[idx]} - Similaridad: {similarities[idx]:.4f}")