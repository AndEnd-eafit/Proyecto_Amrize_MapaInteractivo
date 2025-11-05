import streamlit as st
import numpy as np
import cv2
import pytesseract
from PIL import Image
from keras.models import load_model
from detect_pieces import detectar_piezas
from detect_stars import detectar_estrellas
from classify_pieces import detectar_y_clasificar
from classify_pieces import visualizar_predicciones

# Carga del modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Gestos. Incluye función de OCR")

# Barra lateral con solo el texto
with st.sidebar:
    st.subheader("Usa un modelo entrenado en Teachable Machine para identificar imágenes, e inclusive reconoce texto si lo hay")
    filtro = st.radio("Aplicar Filtro", ('Con Filtro', 'Sin Filtro'))

# Opciones para cámara y carga de archivos en la parte principal
col1, col2 = st.columns(2)

with col1:
    cam_ = st.checkbox("Usar Cámara")

with col2:
    upload_ = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

# Función para aplicar el filtro a la imagen
def apply_filter(image, filtro):
    if filtro == 'Con Filtro':
        return cv2.bitwise_not(image)
    return image

# Función para procesar la imagen para la predicción
def process_image(img):
    # Convertir imagen PIL a formato OpenCV
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    # Aplicar el filtro
    img_cv = apply_filter(img_cv, filtro)
    return img_cv

# Función para normalizar la imagen
def normalize_image(img):
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape != (224, 224, 3):
        st.error(f"La imagen tiene una forma inesperada: {img_array.shape}. Se esperaba (224, 224, 3).")
        return None
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    return normalized_image_array

# Función para realizar la predicción
def predict_image(image_array):
    if image_array.shape == (224, 224, 3) and image_array.dtype == np.float32:
        data[0] = image_array
        prediction = model.predict(data)
        return prediction
    else:
        st.error(f"Array tiene una forma o tipo incorrecto: {image_array.shape}, {image_array.dtype}")
        return None

# Función para extraer texto de la imagen
def extract_text_from_image(image):
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    text = pytesseract.image_to_string(image_cv)
    return text.strip()  # Eliminar espacios en blanco alrededor

# Procesar imagen desde la cámara
if cam_:
    img_file_buffer = st.camera_input("Toma una Foto")
    if img_file_buffer is not None:
        # Leer el buffer de la imagen como una imagen PIL
        img = Image.open(img_file_buffer).convert('RGB')
        
        # Mostrar la imagen original
        st.image(img, caption='Imagen original', use_column_width=True)

    if st.button("Detectar estrellas"):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        estrellas = detectar_estrellas(img_cv)

        if estrellas:
            st.subheader("Estrellas detectadas:")
            for i, estrella in enumerate(estrellas):
                st.write(f"Estrella {i+1}: Color {estrella['color']}, Posición (x={estrella['position']['x']}, y={estrella['position']['y']})")
        else:
            st.write("No se detectaron estrellas.")

    if st.button("Detectar y clasificar piezas"):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        piezas = detectar_y_clasificar(img_cv)

        if piezas:
            st.subheader("Piezas detectadas:")
            for i, pieza in enumerate(piezas):
                st.write(f"Pieza {i+1}: Clase {pieza['clase']} ({pieza['confianza']*100:.1f}%), Color {pieza['color']}, Posición (x={pieza['position']['x']}, y={pieza['position']['y']})")
        else:
            st.write("No se detectaron piezas.")

        # Botón para detectar piezas físicas en la imagen capturada
    if st.button("Detectar piezas en imagen de cámara"):
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        piezas = detectar_piezas(img_cv)

        if piezas:
            st.subheader("Piezas detectadas:")
            for i, pieza in enumerate(piezas):
                st.write(f"Pieza {i+1}: Color {pieza['color']}, Posición (x={pieza['position']['x']}, y={pieza['position']['y']})")
        else:
            st.write("No se detectaron piezas.")
       
        # Preparar la imagen para la predicción
        normalized_image_array = normalize_image(img)
        if normalized_image_array is not None:
            # Realizar predicción
            prediction_result = predict_image(normalized_image_array)
            if prediction_result is not None:
                # Mostrar los resultados de la predicción
                prediction_displayed = False
                if prediction_result[0][0] > 0.5:
                    st.header('Palma, con Probabilidad: ' + str(prediction_result[0][0]))
                    prediction_displayed = True
                if prediction_result[0][1] > 0.5:
                    st.header('Apuntar, con Probabilidad: ' + str(prediction_result[0][1]))
                    prediction_displayed = True
               
                if not prediction_displayed:
                    st.write("Ninguna predicción supera el umbral de 0.5.")

        # Procesar la imagen para la extracción de texto
        img_filtered = process_image(img)
        text = extract_text_from_image(Image.fromarray(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)))
        if text:
            st.write("Texto extraído de la imagen con filtro:", text)
        else:
            st.write("No se pudo extraer texto de la imagen.")

# Procesar imagen desde un archivo subido
if upload_ is not None:
    # Leer el archivo subido como una imagen PIL
    uploaded_image = Image.open(upload_).convert('RGB')
    
    # Mostrar la imagen original
    st.image(uploaded_image, caption='Imagen cargada', use_column_width=True)

    # Botón para detectar piezas físicas en la imagen cargada
    if st.button("Detectar piezas en imagen cargada"):
        img_cv = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)
        piezas = detectar_piezas(img_cv)

        if piezas:
            st.subheader("Piezas detectadas:")
            for i, pieza in enumerate(piezas):
                st.write(f"Pieza {i+1}: Color {pieza['color']}, Posición (x={pieza['position']['x']}, y={pieza['position']['y']})")
        else:
            st.write("No se detectaron piezas.")
    
    # Preparar la imagen para la predicción
    normalized_image_array = normalize_image(uploaded_image)
    if normalized_image_array is not None:
        # Realizar predicción
        prediction_result = predict_image(normalized_image_array)
        if prediction_result is not None:
            # Mostrar los resultados de la predicción
            prediction_displayed = False
            if prediction_result[0][0] > 0.5:
                st.header('Palma, con Probabilidad: ' + str(prediction_result[0][0]))
                prediction_displayed = True
            if prediction_result[0][1] > 0.5:
                st.header('Apuntar, con Probabilidad: ' + str(prediction_result[0][1]))
                prediction_displayed = True
            
            if not prediction_displayed:
                st.write("Ninguna predicción supera el umbral de 0.5.")
    
    # Procesar la imagen para la extracción de texto
    img_filtered = process_image(uploaded_image)
    text = extract_text_from_image(Image.fromarray(cv2.cvtColor(img_filtered, cv2.COLOR_BGR2RGB)))
    if text:
        st.write("Texto extraído de la imagen con filtro:", text)
    else:
        st.write("No se pudo extraer texto de la imagen.")


