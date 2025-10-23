import cv2 as cv
import numpy as np

img = cv.imread('C:\\Users\\villi\\OneDrive\\Escritorio\\uni\\imagenIA\\figura.png', 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

########################### ROJO ###################################

ubb = (0, 60, 60)
uba = (10, 255, 255)
ubb1 = (170, 60, 60)
uba1 = (180, 255, 255)

mascara1 = cv.inRange(hsv, ubb, uba)
mascara2 = cv.inRange(hsv, ubb1, uba1)
mascara = mascara1 + mascara2
resultadoRojo = cv.bitwise_and(img, img, mask=mascara)

altura, ancho = mascara.shape
visitados = np.zeros_like(mascara)

for y in range(altura):
    for x in range(ancho):
        if mascara[y, x] == 255 and visitados[y, x] == 0:
            xs = []
            ys = []
            pila = [(y, x)]
            
            while pila:
                cy, cx = pila.pop()
                if cy < 0 or cy >= altura or cx < 0 or cx >= ancho or visitados[cy, cx] or mascara[cy, cx] == 0:
                    continue
                
                visitados[cy, cx] = 1
                xs.append(cx)
                ys.append(cy)
                
                pila.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
            
            if xs:
                centro_x = (min(xs) + max(xs)) // 2
                centro_y = (min(ys) + max(ys)) // 2
                print(f'{centro_x}, {centro_y}')
                cv.circle(resultadoRojo, (centro_x, centro_y), 7, (0, 255, 0), -1)

########################### VERDE ###################################

ubb = (40, 60, 60)
uba = (80, 255, 255)
mascara1 = cv.inRange(hsv, ubb, uba)
resultadoVerde = cv.bitwise_and(img, img, mask=mascara1)

altura, ancho = mascara1.shape
visitados = np.zeros_like(mascara1)

for y in range(altura):
    for x in range(ancho):
        if mascara1[y, x] == 255 and visitados[y, x] == 0:
            xs = []
            ys = []
            pila = [(y, x)]
            
            while pila:
                cy, cx = pila.pop()
                if cy < 0 or cy >= altura or cx < 0 or cx >= ancho or visitados[cy, cx] or mascara1[cy, cx] == 0:
                    continue
                
                visitados[cy, cx] = 1
                xs.append(cx)
                ys.append(cy)
                
                pila.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
            
            if xs:
                centro_x = (min(xs) + max(xs)) // 2
                centro_y = (min(ys) + max(ys)) // 2
                print(f'{centro_x}, {centro_y}')
                cv.circle(resultadoVerde, (centro_x, centro_y), 7, (0, 255, 0), -1)

########################### AZUL ###################################

ubb = (100, 60, 60)
uba = (130, 255, 255)
mascara1 = cv.inRange(hsv, ubb, uba)
resultadoAzul = cv.bitwise_and(img, img, mask=mascara1)

altura, ancho = mascara1.shape
visitados = np.zeros_like(mascara1)

for y in range(altura):
    for x in range(ancho):
        if mascara1[y, x] == 255 and visitados[y, x] == 0:
            xs = []
            ys = []
            pila = [(y, x)]
            
            while pila:
                cy, cx = pila.pop()
                if cy < 0 or cy >= altura or cx < 0 or cx >= ancho or visitados[cy, cx] or mascara1[cy, cx] == 0:
                    continue
                
                visitados[cy, cx] = 1
                xs.append(cx)
                ys.append(cy)
                
                pila.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
            
            if xs:
                centro_x = (min(xs) + max(xs)) // 2
                centro_y = (min(ys) + max(ys)) // 2
                print(f'{centro_x}, {centro_y}')
                cv.circle(resultadoAzul, (centro_x, centro_y), 7, (0, 255, 0), -1)

########################### AMARILLO ###################################

ubb = (20, 60, 60)
uba = (40, 255, 255)
mascara1 = cv.inRange(hsv, ubb, uba)
resultadoAmarillo = cv.bitwise_and(img, img, mask=mascara1)

altura, ancho = mascara1.shape
visitados = np.zeros_like(mascara1)

for y in range(altura):
    for x in range(ancho):
        if mascara1[y, x] == 255 and visitados[y, x] == 0:
            xs = []
            ys = []
            pila = [(y, x)]
            
            while pila:
                cy, cx = pila.pop()
                if cy < 0 or cy >= altura or cx < 0 or cx >= ancho or visitados[cy, cx] or mascara1[cy, cx] == 0:
                    continue
                
                visitados[cy, cx] = 1
                xs.append(cx)
                ys.append(cy)
                
                pila.extend([(cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)])
            
            if xs:
                centro_x = (min(xs) + max(xs)) // 2
                centro_y = (min(ys) + max(ys)) // 2
                print(f'{centro_x}, {centro_y}')
                cv.circle(resultadoAmarillo, (centro_x, centro_y), 7, (0, 255, 0), -1)

cv.imshow('resultado', resultadoRojo)
cv.imshow('resultadoVerde', resultadoVerde)
cv.imshow('resultadoAzul', resultadoAzul)
cv.imshow('resultadoAmarillo', resultadoAmarillo)

cv.imshow('img', img)

cv.waitKey(0)
cv.destroyAllWindows()

