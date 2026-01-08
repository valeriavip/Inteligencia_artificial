import requests
import time
import json
import certifi 

# Configuración 
SUBREDDIT = "Manzo" 
URL = f"https://www.reddit.com/r/{SUBREDDIT}/new.json"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

posts_ya_vistos = set()
NOMBRE_ARCHIVO = f"reddit_data_{SUBREDDIT}.jsonl"

print(f" Iniciando monitoreo de r/{SUBREDDIT} ---")
print(f" Guardando datos en {NOMBRE_ARCHIVO} ---")

try:
    while True:
        try:
            # Hacer la petición GET a Reddit
            response = requests.get(URL, headers=HEADERS, verify=certifi.where())
            
            response.raise_for_status()

            if response.status_code == 200:
                # Convertir la respuesta de JSON a un diccionario de Python
                data = response.json()
                
                # Iterar sobre los posts
                for post in reversed(data['data']['children']):
                    post_data = post['data']
                    post_id = post_data['id']
                    
                    if post_id not in posts_ya_vistos:
                        cuerpo_texto = post_data.get('selftext', "") 
                        
                        titulo = post_data.get('title', "") 
                        url = post_data.get('url', "")

                        # 5. Guardar los datos en un archivo JSON Lines (.jsonl)
                        data_to_save = {
                            'id': post_id,
                            'titulo': titulo,
                            'cuerpo': cuerpo_texto,
                            'url': url,
                            'subreddit': SUBREDDIT
                        }
                        
                        with open(NOMBRE_ARCHIVO, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(data_to_save) + '\n')
                        
                        print(f"Guardado: {post_id} - {titulo[:50]}...")

                        posts_ya_vistos.add(post_id)

                time.sleep(10) 

        except requests.exceptions.RequestException as e:
            print(f"Error de red o HTTP (¿existe el subreddit?): {e}")
            time.sleep(60) 
        except json.JSONDecodeError:
            print("Error: No se pudo decodificar la respuesta JSON. Posiblemente Reddit está caído.")
            time.sleep(60)

except KeyboardInterrupt:
    print(f"\n--- Monitoreo detenido. Datos guardados en {NOMBRE_ARCHIVO} ---")