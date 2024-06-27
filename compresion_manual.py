import numpy as np
from skimage.util import img_as_ubyte
from tqdm import tqdm
from tkinter import filedialog
import cv2

# Funciones de compresión manual
def next_power_of_2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()

def pad_to_power_of_2(image):
    if len(image.shape) == 2:
        image = image[:, :, None]
    rows, cols, channels = image.shape
    new_rows = 2**np.ceil(np.log2(rows)).astype(int)
    new_cols = 2**np.ceil(np.log2(cols)).astype(int)
    padded_image = np.zeros((new_rows, new_cols, channels))
    padded_image[:rows, :cols, :] = image
    if channels == 1:
        padded_image = padded_image[:, :, 0]
    return padded_image, (rows, cols)

def fft_manual(x):
    N = x.shape[0]
    if N <= 1:
        return x
    even = fft_manual(x[0::2])
    odd = fft_manual(x[1::2])
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    T = factor[:N // 2] * odd
    return np.concatenate([even + T, even - T])

def fft2_manual(image):
    padded_image, original_shape = pad_to_power_of_2(image)
    rows_fft = np.array([fft_manual(row) for row in tqdm(padded_image, desc="FFT Rows")])
    cols_fft = np.array([fft_manual(col) for col in tqdm(rows_fft.T, desc="FFT Columns")]).T
    return cols_fft[:image.shape[0], :image.shape[1]]

def ifft_manual(x):
    x_conj = np.conjugate(x)
    result = fft_manual(x_conj)
    return np.conjugate(result) / x.shape[0]

def ifft2_manual(spectrum):
    rows_ifft = np.array([ifft_manual(row) for row in tqdm(spectrum, desc="IFFT Rows")])
    cols_ifft = np.array([ifft_manual(col) for col in tqdm(rows_ifft.T, desc="IFFT Columns")]).T
    return cols_ifft[:spectrum.shape[0], :spectrum.shape[1]]

def compress_channel_manual(channel, compression_rate):
    transformed = fft2_manual(channel)
    flat = np.abs(transformed).flatten()
    threshold = np.percentile(flat, (1 - compression_rate) * 100)
    transformed[np.abs(transformed) < threshold] = 0
    compressed_channel = ifft2_manual(transformed).real
    return compressed_channel

def compress_image_manual(image, compression_rate):
    padded_image, original_shape = pad_to_power_of_2(image)
    h, w = original_shape
    compressed_image = np.zeros_like(padded_image)
    for channel in range(3):
        compressed_image[..., channel] = compress_channel_manual(padded_image[..., channel], compression_rate)
    return compressed_image[:h, :w, :]

def aplicar_compresion_manual():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            imagen_procesada_cv = compress_image_manual(imagen_procesada_cv, 0.5)
            print(f"Tamaño de la imagen comprimida: {imagen_procesada_cv.shape}")
            print(f"Tipo de datos de la imagen comprimida: {imagen_procesada_cv.dtype}")
            archivo = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp"), ("GIF", "*.gif")],
                title="Guardar imagen comprimida"
            )
            if archivo:
                imagen_comprimida_uint8 = np.clip(imagen_procesada_cv, 0, 255).astype(np.uint8)
                print(f"Tamaño de la imagen antes de guardar: {imagen_comprimida_uint8.shape}")
                print(f"Tipo de datos de la imagen antes de guardar: {imagen_comprimida_uint8.dtype}")
                cv2.imwrite(archivo, cv2.cvtColor(imagen_comprimida_uint8, cv2.COLOR_RGB2BGR))
                print(f"Imagen comprimida guardada como: {archivo}")
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al comprimir la imagen: {e}")
