import os
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from flask_cors import CORS   # <-- tambahkan ini
from gevent.pywsgi import WSGIServer
from io import BytesIO
from PIL import Image

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)   # <-- aktifkan CORS untuk semua route

# Load model
MODEL_PATH = os.path.join("models", "keras_models", "model-mobilenet-RMSprop0.0002-008-0.995584-0.711503.h5")
model = load_model(MODEL_PATH)
print("Model loaded successfully !!")

# Load label makanan
with open(os.path.join("static", "food_list", "food_list.json"), "r", encoding="utf8") as f:
    food_labels = json.load(f)
class_names = sorted(food_labels.keys())
label_dict = dict(zip(range(len(class_names)), class_names))

# Load data kalori
food_calories = pd.read_csv(os.path.join("static", "food_list", "Food_calories.csv"))

# Fungsi preprocessing gambar (langsung dari memory)
def prepare_image(file):
    img = Image.open(BytesIO(file.read())).convert("RGB")
    img = img.resize((224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    return x

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    f = request.files["image"]

    # langsung proses tanpa save
    img = prepare_image(f)
    preds = model.predict(img)
    idx = preds.argmax(axis=-1)[0]
    pred_label = label_dict[idx]
    probability = float(preds.max(axis=-1)[0])

    # Ambil data kalori
    food_retrieve = food_calories[food_calories["name"] == pred_label]
    if food_retrieve.empty:
        return jsonify({
            "label": pred_label,
            "probability": probability,
            "calories": None,
            "unit": None,
            "nutritional_min": None,
            "nutritional_max": None
        })

    response = {
        "label": pred_label,
        "probability": probability,
        "calories": float(food_retrieve["average cal"].values[0]),
        "unit": str(food_retrieve["unit"].values[0]),
        "nutritional_min": float(food_retrieve["nutritional value min,kcal"].values[0]),
        "nutritional_max": float(food_retrieve["nutritional value max,kcal"].values[0])
    }

    return jsonify(response)

if __name__ == "__main__":
    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()
