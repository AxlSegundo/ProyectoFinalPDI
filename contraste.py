import cv2
import numpy as np

def change_contrast(image, contrast_factor):
    # Convertir la imagen a float para evitar problemas de saturación
    image = image.astype(np.float32)
    
    # Ajustar el contraste de la imagen
    adjusted_image = contrast_factor * (image - 128) + 128
    
    # Asegurarse de que los valores se mantienen en el rango [0, 255] manualmente
    adjusted_image[adjusted_image > 255] = 255
    adjusted_image[adjusted_image < 0] = 0
    
    # Convertir la imagen de vuelta a uint8
    return adjusted_image.astype(np.uint8)

# Cargar la imagen usando OpenCV
image_path = 'Test/Calaca.jpg'
image = cv2.imread(image_path)

# Factor de contraste (1.0 significa sin cambio, >1.0 incrementa el contraste, <1.0 disminuye el contraste)
contrast_factor = 2.0 # Por ejemplo, para aumentar el contraste

# Cambiar el contraste de la imagen
contrasted_image = change_contrast(image, contrast_factor)

# Guardar la imagen con el contraste ajustado
cv2.imwrite('Test/calaca_contraste.jpg', contrasted_image)
