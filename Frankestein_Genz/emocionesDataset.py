import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasetTexto.csv') 

# Vectorización (TF-IDF)
vectorizer = TfidfVectorizer(
    min_df=1,               
    ngram_range=(1, 2),     
    strip_accents='unicode'
)

# X son features y target
X = vectorizer.fit_transform(df['Comentario_Reaccion'].astype(str))
y = df['Clasificacion_Final'] 

# Separar en entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Support Vector Classification (SVM)
model = LinearSVC(class_weight='balanced', random_state=42)

# Entrenar
model.fit(X_train, y_train)

# Probar con frases nuevas
nuevas_frases = [
    "La cinematografía es absoluta poesía visual.", 
    "El guion es un poco lento y aburrido."
]

X_nuevas = vectorizer.transform(nuevas_frases)
predicciones = model.predict(X_nuevas)

for frase, prediccion in zip(nuevas_frases, predicciones):
    print(f"'{frase}' -> {prediccion}")