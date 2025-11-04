import cv2
import numpy as np

def detectar_estrellas(imagen_bgr):
    piezas = []

    # Preprocesamiento
    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontrar contornos
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = imagen_bgr.shape[:2]

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        # Aproximar contorno para reducir vértices
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Heurística: estrellas suelen tener entre 8 y 12 vértices
        if 8 <= len(approx) <= 12:
            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2
            roi = imagen_bgr[y:y+h, x:x+w]
            color_promedio = cv2.mean(roi)[:3]
            color_rgb = tuple(int(c) for c in color_promedio[::-1])

            piezas.append({
                "forma": "estrella",
                "color": "#{:02X}{:02X}{:02X}".format(*color_rgb),
                "position": {
                    "x": round(cx / ancho, 3),
                    "y": round(cy / alto, 3)
                }
            })

    return piezas
