import cv2
import numpy as np

def histogram_equalization(image):
    # Asegurarse de que la imagen es en escala de grises
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertir la imagen a escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular el histograma de la imagen
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 255))
    
    # Calcular la función de distribución acumulativa (CDF)
    cdf = histogram.cumsum()
    cdf_normalized = cdf * histogram.max() / cdf.max()  # Normalizar el CDF
    
    # Ecualizar la imagen usando la CDF
    cdf_m = np.ma.masked_equal(cdf, 0)  # Mascarar los valores cero del CDF
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # Normalizar el CDF enmascarado
    cdf_final = np.ma.filled(cdf_m, 0).astype('uint8')  # Llenar los valores enmascarados con ceros y convertir a uint8
    
    # Mapear los valores de la imagen original a los valores ecualizados usando el CDF final
    image_equalized = cdf_final[image]
    
    return image_equalized

# Cargar la imagen usando OpenCV
image_path = 'Test/Calaca.jpg'
image = cv2.imread(image_path)

# Realizar la ecualización del histograma
equalized_image = histogram_equalization(image)

# Guardar la imagen ecualizada
cv2.imwrite('Test/calaca_ecualizada.jpg', equalized_image)
