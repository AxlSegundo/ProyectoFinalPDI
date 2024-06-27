import numpy as np
from scipy.fftpack import fft2, ifft2
from skimage import img_as_float, img_as_ubyte

def pad_to_power_of_two(image):
    """ Pad image dimensions to the next power of two. """
    h, w = image.shape[:2]
    new_h = 1 << (h - 1).bit_length()
    new_w = 1 << (w - 1).bit_length()
    padded_image = np.zeros((new_h, new_w, 3))
    padded_image[:h, :w, :] = image
    return padded_image, (h, w)

def compress_channel(channel, compression_rate):
    """ Apply FFT, zero out coefficients, and apply inverse FFT. """
    # Perform FFT on the 2D channel
    transformed = fft2(channel)
    
    # Flatten the 2D FFT result and sort by magnitude
    flat = np.abs(transformed).flatten()
    threshold = np.percentile(flat, compression_rate * 100)

    # Zero out the smallest coefficients
    transformed[np.abs(transformed) < threshold] = 0

    # Perform inverse FFT
    compressed_channel = ifft2(transformed).real
    return compressed_channel

def compress_image(image, compression_rate):
    """ Compress an image using FFT at the given compression rate. """
    padded_image, original_shape = pad_to_power_of_two(image)
    h, w = original_shape
    compressed_image = np.zeros_like(padded_image)
    
    for channel in range(3):
        compressed_image[..., channel] = compress_channel(padded_image[..., channel], compression_rate)
    
    return compressed_image[:h, :w, :]
