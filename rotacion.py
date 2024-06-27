import cv2
import numpy as np

def rotate_90_clockwise(image):
    height, width, channels = image.shape
    rotated_image = np.zeros((width, height, channels), dtype=image.dtype)

    for i in range(height):
        for j in range(width):
            rotated_image[j, height-i-1] = image[i, j]
    
    return rotated_image

def rotate_90_counterclockwise(image):
    height, width, channels = image.shape
    rotated_image = np.zeros((width, height, channels), dtype=image.dtype)

    for i in range(height):
        for j in range(width):
            rotated_image[width-j-1, i] = image[i, j]
    
    return rotated_image

def rotate_180(image):
    height, width, channels = image.shape
    rotated_image = np.zeros((height, width, channels), dtype=image.dtype)

    for i in range(height):
        for j in range(width):
            rotated_image[height-i-1, width-j-1] = image[i, j]
    
    return rotated_image

if __name__ == "__main__":
    image = cv2.imread("Test/Calaca.jpg")
    rotated_image = rotate_90_clockwise(image)
    cv2.imwrite("Test/calaca_90.jpg", rotated_image)
    rotated_image = rotate_90_counterclockwise(image)
    cv2.imwrite("Test/calaca_270.jpg", rotated_image)
    rotated_image = rotate_180(image)
    cv2.imwrite("Test/calaca_180.jpg", rotated_image)