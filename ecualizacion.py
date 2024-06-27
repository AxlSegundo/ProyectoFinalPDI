import cv2
import numpy as np

def histogram_equalization(image):
    # Asegurarse de que la imagen es en escala de grises
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertir la imagen a escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular el histograma manualmente
    histogram = np.zeros(256, dtype=int)
    for value in image.ravel():
        histogram[value] += 1
    
    # Calcular la función de distribución acumulativa (CDF)
    cdf = np.zeros(256, dtype=int)
    cdf[0] = histogram[0]
    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + histogram[i]
    
    # Encontrar el valor mínimo del CDF que no sea cero
    cdf_min = np.min(cdf[np.nonzero(cdf)])
    
    # Normalizar el CDF manualmente
    cdf_normalized = np.zeros(256, dtype=int)
    for i in range(256):
        cdf_normalized[i] = round((cdf[i] - cdf_min) * 255 / (cdf[-1] - cdf_min))
        if cdf[i] < cdf_min:
            cdf_normalized[i] = 0
    
    # Mapear los valores de la imagen original a los valores ecualizados usando el CDF final
    image_equalized = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image_equalized[i, j] = cdf_normalized[image[i, j]]
    
    return image_equalized

# Cargar la imagen usando OpenCV
image_path = 'Test/Calaca.jpg'
image = cv2.imread(image_path)

# Realizar la ecualización del histograma
equalized_image = histogram_equalization(image)

# Guardar la imagen ecualizada
cv2.imwrite('Test/calaca_ecualizada.jpg', equalized_image)
