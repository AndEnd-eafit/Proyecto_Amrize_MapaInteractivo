from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from classify_pieces import detectar_y_clasificar

app = Flask(__name__)

@app.route('/piezas', methods=['POST'])
def piezas():
    if 'imagen' not in request.files:
        return jsonify({"error": "No se recibió ninguna imagen"}), 400

    file = request.files['imagen']
    img = Image.open(BytesIO(file.read())).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    piezas_detectadas = detectar_y_clasificar(img_cv)
    return jsonify({"pieces": piezas_detectadas})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
