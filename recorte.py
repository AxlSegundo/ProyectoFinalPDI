import cv2
import numpy as np

# Variables globales para almacenar las coordenadas del recorte
ref_point = []
cropping = False

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping

    # Si se presiona el botón izquierdo del mouse, se inicia el recorte
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    # Si se suelta el botón izquierdo del mouse, se completa el recorte
    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False

        # Dibujar un rectángulo en la imagen
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("image", image)

# Cargar la imagen usando OpenCV
image_path = 'Test/Calaca.jpg'
image = cv2.imread(image_path)
clone = image.copy()

cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

# Mantener la ventana abierta hasta que se presione la tecla 'q'
while True:
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    # Si se presiona la tecla 'r', reiniciar la selección
    if key == ord("r"):
        image = clone.copy()

    # Si se presiona la tecla 'c', guardar la imagen recortada
    elif key == ord("c"):
        if len(ref_point) == 2:
            crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
            cv2.imshow("crop_img", crop_img)
            cv2.imwrite('Test/calaca_recorte.jpg', crop_img)
        break

    # Si se presiona la tecla 'q', salir del bucle
    elif key == ord("q"):
        break

cv2.destroyAllWindows()
