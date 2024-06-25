import cv2
import numpy as np

def change_brightness(image, brightness_factor):
    # Asegurarse de que el brillo se mantiene en el rango [0, 255]
    image = np.clip(image + brightness_factor, 0, 255)
    return image.astype(np.uint8)

# Cargar la imagen usando OpenCV
image_path = 'Test/Calaca.jpg'
image = cv2.imread(image_path)

# Factor de brillo (puede ser positivo o negativo)
brightness_factor = 10  # Por ejemplo, para aumentar el brillo en 50 unidades

# Cambiar el brillo de la imagen
brightened_image = change_brightness(image, brightness_factor)

# Guardar la imagen con el brillo ajustado
cv2.imwrite('Test/calaca_brillo.jpg', brightened_image)
