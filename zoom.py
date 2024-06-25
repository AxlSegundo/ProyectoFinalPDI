import cv2
import numpy as np

def bilinear_interpolation(image, new_width, new_height):
    height, width, channels = image.shape
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    x_ratio = width / new_width
    y_ratio = height / new_height

    for i in range(new_height):
        for j in range(new_width):
            x_l = int(np.floor(x_ratio * j))
            x_h = min(x_l + 1, width - 1)
            y_l = int(np.floor(y_ratio * i))
            y_h = min(y_l + 1, height - 1)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_l

            for k in range(channels):
                a = image[y_l, x_l, k]
                b = image[y_l, x_h, k]
                c = image[y_h, x_l, k]
                d = image[y_h, x_h, k]

                pixel_value = (a * (1 - x_weight) * (1 - y_weight) +
                               b * x_weight * (1 - y_weight) +
                               c * (1 - x_weight) * y_weight +
                               d * x_weight * y_weight)

                resized_image[i, j, k] = int(pixel_value)

    return resized_image

