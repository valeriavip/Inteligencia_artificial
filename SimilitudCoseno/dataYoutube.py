import pandas as pd
from youtube_comment_downloader import YoutubeCommentDownloader
from datetime import datetime
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()

archivo_original = 'dataset_limpio.csv'

# Palabras clave para filtrar comentarios
palabras_clave = [
    "modernidad líquida", "sociedad del cansancio", "vacío existencial", "autenticidad", "autonomía", "libertad",
    "ansiedad", "depresión", "burnout", "agotamiento", "soledad", "frustración",  "miedo", "aislamiento",
    "algoritmo", "inteligencia artificial", "ia", "redes sociales", "hiperconectividad","pantallas","viralidad", 
    "influencer", "narcisismo", "burbujas de filtro", "sentido", "futuro"
]

palabras_positivas = [
    "esperanza", "auténtico", "libre", "libertad", "verdad", "real", "consciencia",
    "amor", "conexión", "crecer", "aprender", "bueno", "excelente", "genial", 
    "gracias", "feliz", "vida", "claridad", "luz", "paz", "humano", "mejorar",
    "despertar", "valioso", "bonito", "ayuda", "interesante"
]

palabras_negativas = [
    "ansiedad", "depresión", "miedo", "vacío", "falso", "mentira", "esclavo",
    "triste", "soledad", "solo", "cansancio", "harto", "odio", "mal", "terrible",
    "crisis", "adicción", "perdido", "oscuro", "muerte", "dolor", "frustración",
    "aislamiento", "tóxico", "estrés", "agobio", "pesadilla", "fingir", "robot",
    "control", "manipulación", "basura", "asco", "negativo"
]

# Análisis de sentimiento 
def analizar_sentimiento(texto):
    """Cuenta palabras positivas vs negativas y decide."""
    texto = texto.lower()
    score = 0
    
    for p in palabras_positivas:
        if p in texto:
            score += 1
            
    for n in palabras_negativas:
        if n in texto:
            score -= 1
    
    if score > 0:
        return "positivo"
    elif score < 0:
        return "negativo"
    else:
        return "neutral"

mis_videos = [
    ("https://www.youtube.com/watch?v=OeYZwD13oHk", "Crisis de sentido"),
    ("https://www.youtube.com/watch?v=znq3ql6wqnE", "IA y Autonomía"),
    ("https://www.youtube.com/watch?v=KCAS7zZsyE4", "Identidad liquida"),
    ("https://www.youtube.com/watch?v=9wcmXgLF5sY", "Postmodernidad y Miedo"),
]

# Descargar y filtrar
downloader = YoutubeCommentDownloader()
lista_para_guardar = []
for url, tema in mis_videos:
    print(f"Leyendo comentarios: {url}...")
    try:
        comentarios = downloader.get_comments_from_url(url, sort_by=0)
        
        contador = 0
        for c in comentarios:
            texto = c['text']
        
            if len(texto) > 20: 
                if any(p in texto.lower() for p in palabras_clave):
                    
                    sentimiento_detectado = analizar_sentimiento(texto)
                    
                    lista_para_guardar.append({
                        "fecha": datetime.now().strftime("%Y-%m-%d"),
                        "usuario": f"yt_user_{c['author'][:10]}", 
                        "texto": texto,
                        "tema": tema,
                        "sentimiento": sentimiento_detectado, 
                        "likes": c.get('votes', 0),
                        "reposts": 0
                    })
                    contador += 1
            
            if contador >= 500:
                break
                
        print(f"Se guardaron {contador} comentarios.")
                
    except Exception as e:
        print(f"Error en video: {e}")

# Guardar en CSV
if lista_para_guardar:
    df_nuevo = pd.DataFrame(lista_para_guardar)
    
    if os.path.exists(archivo_original):
        df_viejo = pd.read_csv(archivo_original)
        ultimo_id = df_viejo['id'].max()
    else:
        df_viejo = pd.DataFrame()
        ultimo_id = 0
        
    df_nuevo.insert(0, 'id', range(ultimo_id + 1, ultimo_id + 1 + len(df_nuevo)))
    
    df_final = pd.concat([df_viejo, df_nuevo], ignore_index=True)
    df_final.to_csv('dataset_youtube.csv', index=False)
    print(f"\n Se agregaron {len(df_nuevo)} comentarios al archivo 'dataset_youtube.csv'.")
else:
    print("\nNo se encontraron comentarios relevantes")