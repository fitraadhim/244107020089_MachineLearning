import pickle
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, render_template_string, request
from skimage.feature import hog


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "day_night_model.pkl"
SCALER_PATH = BASE_DIR / "scaler.pkl"

app = Flask(__name__)


def load_artifacts():
    with MODEL_PATH.open("rb") as model_file:
        model = pickle.load(model_file)
    with SCALER_PATH.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler


model, scaler = load_artifacts()


def preprocess_image(image_bytes):
    image_array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File yang diunggah bukan gambar yang valid.")

    image = cv2.resize(image, (256, 256))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return scaler.transform(hog_features.reshape(1, -1))


PAGE_TEMPLATE = """
<!doctype html>
<html lang="id">
  <head><meta charset="utf-8"><title>Day vs Night Classifier</title></head>
  <body style="font-family: sans-serif; max-width: 640px; margin: 3rem auto; text-align: center;">
    <h1>Day vs Night Classifier</h1>
    {% if result %}<h2>Hasil: {{ result }}</h2>{% endif %}
    {% if error %}<p style="color: darkred;">{{ error }}</p>{% endif %}
    <form action="/predict" method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*" required>
      <button type="submit">Prediksi</button>
    </form>
  </body>
</html>
"""


@app.get("/")
def home():
    return render_template_string(PAGE_TEMPLATE, result=None, error=None)


@app.post("/predict")
def predict():
    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return render_template_string(
            PAGE_TEMPLATE, result=None, error="Pilih file gambar terlebih dahulu."
        ), 400

    try:
        features = preprocess_image(uploaded_file.read())
        prediction = int(model.predict(features)[0])
        label = "Day (Siang)" if prediction == 1 else "Night (Malam)"
        return render_template_string(PAGE_TEMPLATE, result=label, error=None)
    except (OSError, ValueError) as error:
        return render_template_string(
            PAGE_TEMPLATE, result=None, error=f"Error: {error}"
        ), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
