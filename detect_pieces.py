import cv2
import numpy as np

def detectar_piezas(imagen_bgr):
    piezas = []

    # Convertir a escala de grises y aplicar desenfoque
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)

    # Umbral binario para separar las piezas del fondo
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    alto, ancho, _ = imagen_bgr.shape

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignorar ruido o piezas muy pequeñas
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        # Extraer región de interés (ROI)
        roi = imagen_bgr[y:y+h, x:x+w]

        # Calcular color dominante (promedio)
        color_promedio = cv2.mean(roi)[:3]  # Ignorar canal alfa si existe
        color_rgb = tuple(int(c) for c in color_promedio[::-1])  # BGR → RGB

        pieza = {
            "color": "#{:02X}{:02X}{:02X}".format(*color_rgb),
            "position": {
                "x": round(cx / ancho, 3),
                "y": round(cy / alto, 3)
            }
        }
        piezas.append(pieza)

    return piezas
