import cv2 as cv

img = cv.imread('C:\\Users\\villi\\OneDrive\\Escritorio\\uni\\imagenIA\\ejemplo3.jpg', 1)
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

ubb = (0, 60, 60)
uba = (10, 255, 255)
ubb1 = (170, 60, 60)
uba1 = (180, 255, 255)

mascara1 = cv.inRange(hsv, ubb, uba)
mascara2 = cv.inRange(hsv, ubb1, uba1)

mascara = mascara1 + mascara2
resultado = cv.bitwise_and(img, img, mask=mascara)

cv.imshow('resultado', resultado)
cv.imshow('mascara1', mascara1)
cv.imshow('mascara2', mascara2)
cv.imshow('img', img)
#cv.imshow('img2', img2)
#cv.imshow('hsv', hsv)


cv.waitKey(0)
cv.destroyAllWindows()