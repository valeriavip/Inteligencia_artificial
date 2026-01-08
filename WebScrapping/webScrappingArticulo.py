import fitz  
import os

carpeta_pdfs = "C:\\Users\\villi\\OneDrive\\Escritorio\\WebScrapping\\pdf"      
palabras_clave = [
    "modernidad líquida", "sociedad del cansancio", "vacío existencial", "autenticidad", "autonomía", "libertad",
    "bauman", "byung-chul han", "foucault", "sartre", "camus", "heidegger", "lyotard", "habermas",
    "ansiedad", "depresión", "burnout", "agotamiento", "soledad", "frustración",  "miedo", "aislamiento",
    "algoritmo", "inteligencia artificial", "ia", "redes sociales", "hiperconectividad","pantallas","viralidad", "influencer", "narcisismo", "burbujas de filtro",
]
archivo_resultado = "resumen_filtrado.txt"

with open(archivo_resultado, "w", encoding="utf-8") as archivo_final:
    if os.path.exists(carpeta_pdfs):
        archivos = [f for f in os.listdir(carpeta_pdfs) if f.endswith(".pdf")]
    else:
        print(f"Error: Crea la carpeta '{carpeta_pdfs}' ")
        archivos = []

    for nombre_pdf in archivos:
        ruta = os.path.join(carpeta_pdfs, nombre_pdf)
        print(f"Analizando: {nombre_pdf}...")

        try:
            doc = fitz.open(ruta)
            texto_completo = ""
            for pagina in doc:
                texto_completo += pagina.get_text() + "\n"
            
            #chunks
            chunks = []
            for i in range(0, len(texto_completo), 1000):
                chunks.append(texto_completo[i : i + 1000])

            encontrados = 0
            for chunk in chunks:
                texto_minuscula = chunk.lower()
                
                #palabra clave
                tiene_palabra_clave = False
                for palabra in palabras_clave:
                    if palabra.lower() in texto_minuscula:
                        tiene_palabra_clave = True
                        break 


                #guardar
                if tiene_palabra_clave:
                    separador = f"\n--- HALLAZGO EN: {nombre_pdf} ---\n"
                    archivo_final.write(separador + chunk + "\n")
                    encontrados += 1

            print(f" Se guardaron {encontrados} chunks.")

        except Exception as e:
            print(f"Error leyendo este archivo: {e}")

print(f"\n Archivo:'{archivo_resultado}'")
