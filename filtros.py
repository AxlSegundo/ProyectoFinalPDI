import cv2
import numpy as np

def convert_grayscale(image):
    height, width, channels = image.shape
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            grayscale_image[i, j] = 0.299*r + 0.587*g + 0.114*b

    return grayscale_image

def black_and_white(image, threshold=128):
    image = convert_grayscale(image)
    height, width = image.shape
    bw_image = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            if image[i, j] > threshold:
                bw_image[i, j] = 255
            else:
                bw_image[i, j] = 0
    
    return bw_image

def apply_sepia_filter_manual(image):
    height, width, channels = image.shape
    sepia_image = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            tr = 0.393 * r + 0.769 * g + 0.189 * b
            tg = 0.349 * r + 0.686 * g + 0.168 * b
            tb = 0.272 * r + 0.534 * g + 0.131 * b
            
            sepia_image[i, j, 2] = min(tr, 255)  # Red channel
            sepia_image[i, j, 1] = min(tg, 255)  # Green channel
            sepia_image[i, j, 0] = min(tb, 255)  # Blue channel
    
    return sepia_image.astype(np.uint8)

def apply_color_filter(image, color):
    grayscale_image = convert_grayscale(image)
    height, width = grayscale_image.shape
    colored_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            gray_value = grayscale_image[i, j]
            colored_image[i, j] = [gray_value * (color[0] / 255.0), 
                                   gray_value * (color[1] / 255.0), 
                                   gray_value * (color[2] / 255.0)]
    return colored_image

def apply_pixelate_filter(image, pixel_size):
    height, width, channels = image.shape
    pixelated_image = np.zeros_like(image)

    for y in range(0, height, pixel_size):
        for x in range(0, width, pixel_size):
            # Definir el área del bloque de píxeles
            y_end = min(y + pixel_size, height)
            x_end = min(x + pixel_size, width)
            
            # Obtener el bloque de píxeles
            block = image[y:y_end, x:x_end]
            
            # Calcular el color promedio del bloque
            avg_color = block.mean(axis=(0, 1)).astype(int)
            
            # Asignar el color promedio al bloque
            pixelated_image[y:y_end, x:x_end] = avg_color

    return pixelated_image

def apply_blur_filter(image, kernel_size):
    height, width, channels = image.shape
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    blurred_image = np.zeros_like(image)
    
    for i in range(height):
        for j in range(width):
            for k in range(channels):
                blurred_image[i, j, k] = np.mean(padded_image[i:i+kernel_size, j:j+kernel_size, k])
    
    return blurred_image

def apply_sobel_filter(image):
    grayscale_image = convert_grayscale(image)
    height, width = grayscale_image.shape
    sobel_image = np.zeros((height, width), dtype=np.uint8)

    # Máscaras de Sobel
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Aplicar padding
    padded_image = np.pad(grayscale_image, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            gx = np.sum(sobel_x * padded_image[i-1:i+2, j-1:j+2])
            gy = np.sum(sobel_y * padded_image[i-1:i+2, j-1:j+2])
            sobel_image[i-1, j-1] = min(np.sqrt(gx**2 + gy**2), 255)
    
    return sobel_image

if __name__ == "__main__":
    # image = cv2.imread("Test/Bonsai-Plant.png")
    # grayscale_image = convert_grayscale(image)
    # cv2.imwrite("Test/bonsai_grayscale.png", grayscale_image)
    # bw_image = black_and_white(image)
    # cv2.imwrite("Test/bonsai_bw.jpg", bw_image)
    new_image = cv2.imread("Test/Calaca.jpg")
    toaster_image = apply_sepia_filter_manual(new_image)
    cv2.imwrite("Test/calaca_sepia.jpg", toaster_image)
    # colored_image = apply_color_filter(new_image, [30, 255, 0])
    # cv2.imwrite("Test/bonsai_green.png", colored_image)
    # colored_image = apply_color_filter(new_image, [0, 255, 255])
    # cv2.imwrite("Test/bonsai_yellow.png", colored_image)
    # colored_image = apply_color_filter(new_image, [255, 0, 255])
    # cv2.imwrite("Test/bonsai_purple.png", colored_image)
    # colored_image = apply_color_filter(new_image, [255, 255, 0])
    # cv2.imwrite("Test/bonsai_cyan.png", colored_image)
    # pixelated_image = apply_pixelate_filter(new_image, 20)
    # cv2.imwrite("Test/bonsai_pixelated.png", pixelated_image)
    # blurred_image = apply_blur_filter(new_image, 20)
    # cv2.imwrite("Test/bonsai_blurred.png", blurred_image)
    # sobel_image = apply_sobel_filter(new_image)
    # cv2.imwrite("Test/calaca_sobel.jpg", sobel_image)
