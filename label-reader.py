# Emiliano Nuñez Felix A01645413
# Sergio Santiago Sánchez Salazar A01645255
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# Cargamos la imagen en escala de grises.
imagen = cv2.imread('milk.jpg', cv2.IMREAD_GRAYSCALE)

# Invertimos la imagen para un mejor contraste.
imagen_invertida = cv2.bitwise_not(imagen)

# Aplicar umbralización, los píxeles mayores a 180 se vuelven blancos, los menores negros.
_, umbralizada = cv2.threshold(imagen_invertida, 180, 255, cv2.THRESH_BINARY)

# Definimos el kernel de 3x3
kernel = np.ones((3, 3), np.uint8)
# Eliminamos ruido pequeño.
apertura = cv2.morphologyEx(umbralizada, cv2.MORPH_OPEN, kernel)
# Unimos las letras separadas.
cierre = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

