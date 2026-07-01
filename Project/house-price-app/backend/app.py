from flask import Flask, render_template, request, jsonify
import joblib
import json
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "house_price_model.pkl")
META_PATH = os.path.join(BASE_DIR, "..", "model", "metadata.json")

app = Flask(__name__)

model = joblib.load(MODEL_PATH)
with open(META_PATH) as f:
    metadata = json.load(f)

@app.route("/")
def home():
    return render_template(
        "index.html",
        locations=metadata["locations"],
        furnishing_options=metadata["furnishing_options"],
        model_name=metadata["best_model_name"],
        r2_score=metadata["r2_score"],
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([{
            "location": data["location"],
            "total_sqft": float(data["total_sqft"]),
            "bhk": int(data["bhk"]),
            "bath": int(data["bath"]),
            "balcony": int(data["balcony"]),
            "age_years": float(data["age_years"]),
            "furnishing": data["furnishing"],
            "parking": int(data["parking"]),
        }])
        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 0)
        return jsonify({
            "success": True,
            "price_lakhs": round(float(prediction), 2),
            "price_inr": round(float(prediction) * 100000, 0),
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)