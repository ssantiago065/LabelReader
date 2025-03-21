# Emiliano Nuñez Felix A01645413
# Sergio Santiago Sánchez Salazar A01645255
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# Cargamos la imagen en escala de grises.
imagen = cv2.imread('milk_tag.jpg', cv2.IMREAD_GRAYSCALE)

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

# Inicializar el lector de OCR y definir los idiomas del lector
reader = easyocr.Reader(['es', 'en']) 
resultados = reader.readtext(cierre)

# Imprimir el texto detectado
print("\n=== TEXTO DETECTADO ===")
for bbox, text, prob in resultados:
    print(f"Texto: {text} | Confianza: {prob:.2f}")

# Mostrar las imágenes procesadas utilizando graficas de matplotlib y etiquetar cada una
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(imagen, cmap='gray')
plt.title('Imagen Original')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(umbralizada, cmap='gray')
plt.title('Umbralización')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cierre, cmap='gray')
plt.title('Procesamiento para OCR')
plt.axis('off')

plt.tight_layout()
plt.show()
