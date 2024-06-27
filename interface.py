import tkinter as tk
from tkinter import ttk, filedialog, Scale, HORIZONTAL
from PIL import Image, ImageTk
import cv2
import numpy as np
from filtros import (apply_sepia_filter_manual, black_and_white, apply_color_filter,
                     apply_pixelate_filter, apply_blur_filter, apply_sobel_filter)
from rotacion import rotate_90_clockwise, rotate_90_counterclockwise, rotate_180
from brillo import change_brightness
from contraste import change_contrast
from zoom import bilinear_interpolation
from compresion import compress_image
from compresion_manual import (compress_image_manual)

# Variables globales para almacenar la imagen original y la procesada en formato OpenCV
imagen_original_cv = None
imagen_procesada_cv = None
rect_start = None
rect_end = None
cropping = False

def cv2_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

def actualizar_imagen_tk(image_cv):
    imagen_pil = cv2_to_pil(image_cv)
    img_tk = ImageTk.PhotoImage(imagen_pil)
    etiqueta_imagen.config(image=img_tk)
    etiqueta_imagen.image = img_tk

def seleccionar_imagen():
    global imagen_original_cv, imagen_procesada_cv
    archivo = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("JPEG", "*.jpg;*.jpeg"), 
                   ("PNG", "*.png"), 
                   ("BMP", "*.bmp"), 
                   ("GIF", "*.gif"),
                   ("Todos los archivos", "*.*")]
    )
    
    if archivo:
        try:
            print(f"Archivo seleccionado: {archivo}")
            imagen_original_cv = cv2.imread(archivo)
            imagen_original_cv = cv2.cvtColor(imagen_original_cv, cv2.COLOR_BGR2RGB)
            imagen_original_cv = cv2.resize(imagen_original_cv, (800, 600))
            imagen_procesada_cv = imagen_original_cv.copy()
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al abrir la imagen: {e}")

def aplicar_filtro(filtro):
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            imagen_procesada_cv = filtro(imagen_procesada_cv)
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al aplicar el filtro: {e}")

def aplicar_sobel_filter_rgb(image):
    sobel_image = apply_sobel_filter(image)
    sobel_image_rgb = cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2RGB)
    return sobel_image_rgb

def aplicar_bn_rgb(image):
    bn_image = black_and_white(image)
    bn_image_rgb = cv2.cvtColor(bn_image, cv2.COLOR_GRAY2RGB)
    return bn_image_rgb

def aplicar_brightness():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            brightness_factor = slider_brightness.get()
            imagen_procesada_cv = change_brightness(imagen_procesada_cv, brightness_factor)
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al cambiar el brillo: {e}")

def aplicar_contrast():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            contrast_factor = slider_contrast.get()
            imagen_procesada_cv = change_contrast(imagen_procesada_cv, contrast_factor)
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al cambiar el contraste: {e}")

def aplicar_resize():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            new_width = int(entry_width.get())
            new_height = int(entry_height.get())
            imagen_procesada_cv = bilinear_interpolation(imagen_procesada_cv, new_width, new_height)
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al cambiar el tamaño: {e}")

def aplicar_rotacion(filtro):
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            imagen_procesada_cv = filtro(imagen_procesada_cv)
            actualizar_imagen_tk(imagen_procesada_cv)
        except Exception as e:
            print(f"Error al aplicar la rotación: {e}")

def start_crop(event):
    global rect_start, cropping
    rect_start = (event.x, event.y)
    cropping = True

def end_crop(event):
    global rect_start, rect_end, cropping, imagen_procesada_cv
    rect_end = (event.x, event.y)
    cropping = False

    if rect_start and rect_end:
        x1, y1 = rect_start
        x2, y2 = rect_end
        if x1 < x2 and y1 < y2:
            cropped_image = imagen_procesada_cv[y1:y2, x1:x2]
            imagen_procesada_cv = cropped_image
            actualizar_imagen_tk(imagen_procesada_cv)
            rect_start = None
            rect_end = None

def draw_crop_rectangle(event):
    global rect_start, rect_end, cropping, imagen_procesada_cv
    if cropping:
        rect_end = (event.x, event.y)
        image_copy = imagen_procesada_cv.copy()
        cv2.rectangle(image_copy, rect_start, rect_end, (0, 255, 0), 2)
        actualizar_imagen_tk(image_copy)

def guardar_imagen():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        archivo = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("BMP", "*.bmp"), ("GIF", "*.gif")],
            title="Guardar imagen"
        )
        if archivo:
            try:
                cv2.imwrite(archivo, cv2.cvtColor(imagen_procesada_cv, cv2.COLOR_RGB2BGR))
                print(f"Imagen guardada como: {archivo}")
            except Exception as e:
                print(f"Error al guardar la imagen: {e}")

def aplicar_compresion():
    global imagen_procesada_cv
    if imagen_procesada_cv is not None:
        try:
            imagen_procesada_cv = compress_image(imagen_procesada_cv, 0.5)
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

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Editor de Imágenes")
ventana.attributes('-fullscreen', True)  # Pantalla completa

frame_superior = tk.Frame(ventana)
frame_superior.pack(side=tk.TOP, fill=tk.X)

# Crear el botón de cierre
boton_cerrar = tk.Button(frame_superior, text="Cerrar", command=ventana.destroy)
boton_cerrar.pack(side=tk.RIGHT, padx=10, pady=10)
# Crear un contenedor de pestañas
notebook = ttk.Notebook(ventana)
notebook.pack(fill='both', expand=True)

# Crear el frame para la imagen y añadirlo al notebook
frame_imagen = ttk.Frame(notebook)
notebook.add(frame_imagen, text='Imagen')

# Crear un contenedor para la imagen
etiqueta_imagen = tk.Label(frame_imagen)
etiqueta_imagen.pack(expand=True)

# Vincular eventos de ratón a la etiqueta de imagen
etiqueta_imagen.bind("<ButtonPress-1>", start_crop)
etiqueta_imagen.bind("<ButtonRelease-1>", end_crop)
etiqueta_imagen.bind("<B1-Motion>", draw_crop_rectangle)

# Crear el frame para filtros y añadirlo al notebook
frame_filtros = ttk.Frame(notebook)
notebook.add(frame_filtros, text='Filtros')

# Añadir botones de filtros al frame de filtros
boton_filtro_sepia = tk.Button(frame_filtros, text="Aplicar Filtro Sepia", command=lambda: aplicar_filtro(apply_sepia_filter_manual))
boton_filtro_sepia.pack(pady=5)

boton_filtro_bn = tk.Button(frame_filtros, text="Blanco y Negro", command=lambda: aplicar_filtro(aplicar_bn_rgb))
boton_filtro_bn.pack(pady=5)

boton_filtro_verde = tk.Button(frame_filtros, text="Filtro Verde", command=lambda: aplicar_filtro(lambda img: apply_color_filter(img, [0, 255, 0])))
boton_filtro_verde.pack(pady=5)

boton_filtro_amarillo = tk.Button(frame_filtros, text="Filtro Amarillo", command=lambda: aplicar_filtro(lambda img: apply_color_filter(img, [255, 255, 0])))
boton_filtro_amarillo.pack(pady=5)

boton_filtro_morado = tk.Button(frame_filtros, text="Filtro Morado", command=lambda: aplicar_filtro(lambda img: apply_color_filter(img, [255, 0, 255])))
boton_filtro_morado.pack(pady=5)

boton_filtro_cian = tk.Button(frame_filtros, text="Filtro Cian", command=lambda: aplicar_filtro(lambda img: apply_color_filter(img, [0, 255, 255])))
boton_filtro_cian.pack(pady=5)

boton_filtro_pixel = tk.Button(frame_filtros, text="Filtro Pixelado", command=lambda: aplicar_filtro(lambda img: apply_pixelate_filter(img, 10)))
boton_filtro_pixel.pack(pady=5)

boton_filtro_blur = tk.Button(frame_filtros, text="Filtro Difuminado", command=lambda: aplicar_filtro(lambda img: apply_blur_filter(img, 10)))
boton_filtro_blur.pack(pady=5)

boton_filtro_sobel = tk.Button(frame_filtros, text="Filtro Sobel", command=lambda: aplicar_filtro(aplicar_sobel_filter_rgb))
boton_filtro_sobel.pack(pady=5)

# Crear el frame para ajustes y añadirlo al notebook
frame_ajustes = ttk.Frame(notebook)
notebook.add(frame_ajustes, text='Ajustes')

# Añadir sliders y botones de brillo y contraste al frame de ajustes
slider_brightness = Scale(frame_ajustes, from_=-100, to=100, orient=HORIZONTAL, label="Ajuste de Brillo")
slider_brightness.pack(pady=5)
boton_brightness = tk.Button(frame_ajustes, text="Aplicar Brillo", command=aplicar_brightness)
boton_brightness.pack(pady=5)

slider_contrast = Scale(frame_ajustes, from_=0.5, to=3.0, resolution=0.1, orient=HORIZONTAL, label="Ajuste de Contraste")
slider_contrast.pack(pady=5)
boton_contrast = tk.Button(frame_ajustes, text="Aplicar Contraste", command=aplicar_contrast)
boton_contrast.pack(pady=5)

# Añadir botones de rotación al frame de ajustes
boton_rotar_90 = tk.Button(frame_ajustes, text="Rotar 90°", command=lambda: aplicar_rotacion(rotate_90_clockwise))
boton_rotar_90.pack(pady=5)

boton_rotar_90_contrario = tk.Button(frame_ajustes, text="Rotar 90° Contra", command=lambda: aplicar_rotacion(rotate_90_counterclockwise))
boton_rotar_90_contrario.pack(pady=5)

boton_rotar_180 = tk.Button(frame_ajustes, text="Rotar 180°", command=lambda: aplicar_rotacion(rotate_180))
boton_rotar_180.pack(pady=5)

# Crear el frame para tamaño y añadirlo al notebook
frame_tamano = ttk.Frame(notebook)
notebook.add(frame_tamano, text='Tamaño')

# Añadir entradas de texto y botón de tamaño al frame de tamaño
label_width = tk.Label(frame_tamano, text="Ancho:")
label_width.pack(side=tk.LEFT, padx=5, pady=5)
entry_width = tk.Entry(frame_tamano, width=5)
entry_width.pack(side=tk.LEFT, padx=5, pady=5)

label_height = tk.Label(frame_tamano, text="Alto:")
label_height.pack(side=tk.LEFT, padx=5, pady=5)
entry_height = tk.Entry(frame_tamano, width=5)
entry_height.pack(side=tk.LEFT, padx=5, pady=5)

boton_resize = tk.Button(frame_tamano, text="Aplicar Tamaño", command=aplicar_resize)
boton_resize.pack(side=tk.LEFT, padx=5, pady=5)

# Crear el frame para compresión y añadirlo al notebook
frame_compresion = ttk.Frame(notebook)
notebook.add(frame_compresion, text='Compresión')

# Añadir botones de compresión al frame de compresión
boton_comprimir = tk.Button(frame_compresion, text="Compresión Libreria", command=aplicar_compresion)
boton_comprimir.pack(pady=5)

boton_comprimir_manual = tk.Button(frame_compresion, text="Compresión Manual", command=aplicar_compresion_manual)
boton_comprimir_manual.pack(pady=5)

# Crear un botón para seleccionar la imagen y añadirlo al frame de imagen
boton_seleccionar = tk.Button(frame_imagen, text="Seleccionar Imagen", command=seleccionar_imagen)
boton_seleccionar.pack(pady=10)

# Crear un botón para guardar la imagen procesada y añadirlo al frame de imagen
boton_guardar = tk.Button(frame_imagen, text="Guardar Imagen", command=guardar_imagen)
boton_guardar.pack(pady=10)


# Iniciar el bucle principal de la interfaz
ventana.mainloop()
