import pickle
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "sleep_disorder_model.pkl"

with MODEL_PATH.open("rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

FORM_TEMPLATE = """
<!doctype html>
<html lang="id">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Sleep Wise</title>
    <style>
      body { font-family: sans-serif; max-width: 760px; margin: 2rem auto; padding: 0 1rem; }
      form { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
      label { display: grid; gap: .35rem; }
      input, select, button { padding: .6rem; font-size: 1rem; }
      button { grid-column: 1 / -1; cursor: pointer; }
      .result { margin: 1rem 0; padding: 1rem; background: #e8f5e9; }
      .error { margin: 1rem 0; color: #b71c1c; }
    </style>
  </head>
  <body>
    <h1>Sleep Wise</h1>
    <p>Prediksi gangguan tidur berdasarkan data kesehatan dan gaya hidup.</p>
    {% if result %}<div class="result"><strong>Hasil prediksi: {{ result }}</strong></div>{% endif %}
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
    <form action="/predict" method="post">
      <label>Gender
        <select name="Gender"><option>Male</option><option>Female</option></select>
      </label>
      <label>Age <input name="Age" type="number" min="1" required></label>
      <label>Occupation <input name="Occupation" required></label>
      <label>Sleep Duration <input name="Sleep Duration" type="number" step="0.1" required></label>
      <label>Quality of Sleep <input name="Quality of Sleep" type="number" min="1" max="10" required></label>
      <label>Physical Activity Level <input name="Physical Activity Level" type="number" min="0" required></label>
      <label>Stress Level <input name="Stress Level" type="number" min="1" max="10" required></label>
      <label>BMI Category
        <select name="BMI Category"><option>Normal</option><option>Normal Weight</option><option>Overweight</option><option>Obese</option></select>
      </label>
      <label>Heart Rate <input name="Heart Rate" type="number" min="1" required></label>
      <label>Daily Steps <input name="Daily Steps" type="number" min="0" required></label>
      <label>Systolic BP <input name="Systolic BP" type="number" min="1" required></label>
      <label>Diastolic BP <input name="Diastolic BP" type="number" min="1" required></label>
      <button type="submit">Prediksi Gangguan Tidur</button>
    </form>
  </body>
</html>
"""


def home():
    return render_template_string(FORM_TEMPLATE, result=None, error=None)


app.add_url_rule("/", "home", home, methods=["GET"])


@app.post("/predict")
def predict():
    try:
        numeric_fields = [
            "Age", "Sleep Duration", "Quality of Sleep",
            "Physical Activity Level", "Stress Level", "Heart Rate",
            "Daily Steps", "Systolic BP", "Diastolic BP",
        ]
        values = {field: request.form[field] for field in request.form}
        for field in numeric_fields:
            values[field] = float(values[field])

        input_data = pd.DataFrame([values])
        prediction = model.predict(input_data)[0]
        return render_template_string(
            FORM_TEMPLATE, result=str(prediction), error=None
        )
    except (KeyError, TypeError, ValueError) as error:
        return render_template_string(
            FORM_TEMPLATE, result=None, error=f"Data tidak valid: {error}"
        ), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
