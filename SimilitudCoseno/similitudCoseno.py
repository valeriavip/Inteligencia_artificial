from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import numpy as np
import nltk
import pandas as pd

vec1 = np.array([[1,1,0,1,1]])
vec2 = np.array([[0,1,0,1,1]])

# Descargar la lista de palabras vacías
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Obtener la lista de stopwords 
lista_stopwords = stopwords.words('spanish')

print(cosine_similarity(vec1, vec2))

#Defenimos la data
tw1 = "Realizarán patrullajes de vigilancia en distintos puntos de la ciudad; se quedarán el tiempo que sea necesario hasta que la tranquilidad se restablezca" 
nt1 = "Con el objetivo de reforzar la protección a la población civil, un contingente de 230 elementos de la Secretaría de la Defensa Nacional (Sedena) arribó la noche de este viernes a Culiacán, luego del violento episodio ocurrido el pasado jueves 17 de octubre." 
tw2 = "Personal del #EjércitoyFAM perteneciente al 22/o. Batallón de Infantería en coordinación con autoridades municipales llevaron a cabo la campaña “Canje de Armas de Fuego por Vales de Despensa” en el municipio de Los Aldama," 
tw3 = "La @SEDENAmx despliega a 230 elementos para reforzar la seguridad en Culiacán, luego de la jornada violenta que terminó en la liberación de Ovidio Guzmán, uno de los hijos de El Chapo."
tw4 = "Jupyter Notebook is an open-source, interactive web application that allows you to write and run "
twits = [tw1, nt1, tw2, tw3, tw4] 


# Crear tabla de frecuencias
count_vectorizer = CountVectorizer(stop_words=lista_stopwords)
#count_vectorizer = CountVectorizer()
matrix = count_vectorizer.fit_transform(twits)


# utilizacion de pandas para generar la matriz de frecuencias 
data_matrix = matrix.todense()
frecuencias = pd.DataFrame(data_matrix, 
                  columns=count_vectorizer.get_feature_names_out(), 
                  index=['tw1', 'nt1', 'tw2', 'tw3', 'tw4'])

frecuencias.loc["tw1":"tw2", "17":"batallon"]

frecuencias.loc['tw1',:].head(40)
salida = cosine_similarity(frecuencias, frecuencias)
count_vectorizer.get_feature_names_out()
count_vectorizer.vocabulary_
frecuencias.columns.tolist()
frecuencias.iloc[2]['aldama']

import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))
plt.imshow(salida)
plt.colorbar()

plt.ylabel(['tw1', 'nt1', 'tw2', 'tw3', 'tw4'])
plt.show()

