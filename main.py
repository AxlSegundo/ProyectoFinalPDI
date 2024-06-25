import cv2
import numpy as np
from zoom import bilinear_interpolation

path='Test/Calaca.jpg'
image = cv2.imread(path)
H,W,C=image.shape

print("Ampliar o reducir tam")
print(f"El ancho de tu imagen es {W} pixeles y la altura es de {H} pixeles")
Ancho = int(input("Dime la nueva anchura de la imagen: "))
Alto = int(input("Dime la nueva altura de la imagen: "))
resized= bilinear_interpolation(image, Ancho, Alto)
# Guardar la imagen redimensionada
cv2.imwrite('Test/calaca_redimensionada.jpg', resized)
print("El proceso ha terminado")