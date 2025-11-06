import cv2
import numpy as np
from PIL import Image
from keras.models import load_model

# Cargar el modelo
model = load_model('keras_model.h5')

# Cargar etiquetas
with open('labels.txt') as f:
    labels = [line.strip().split(" ")[1] for line in f.readlines()]

# Normalizar imagen
def normalize_image(img_pil):
    img = img_pil.resize((224, 224))
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    normalized = (img_array.astype(np.float32) / 127.0) - 1
    return normalized

# Clasificar una región
def classify_roi(roi_bgr):
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(roi_rgb)
    normalized = normalize_image(img_pil)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized
    prediction = model.predict(data)
    index = np.argmax(prediction[0])
    label = labels[index]
    confidence = float(prediction[0][index])
    return label, confidence

# Detectar y clasificar piezas
def detectar_y_clasificar(imagen_bgr):
    piezas = []

    gris = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    _, binaria = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    alto, ancho = imagen_bgr.shape[:2]

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2
        roi = imagen_bgr[y:y+h, x:x+w]

        # Color promedio
        color_promedio = cv2.mean(roi)[:3]
        color_rgb = tuple(int(c) for c in color_promedio[::-1])

        # Clasificación
        label, confidence = classify_roi(roi)

        piezas.append({
            "clase": label,
            "confianza": round(confidence, 2),
            "color": "#{:02X}{:02X}{:02X}".format(*color_rgb),
            "position": {
                "x": round(cx / ancho, 3),
                "y": round(cy / alto, 3)
            }
        })

    return piezas

