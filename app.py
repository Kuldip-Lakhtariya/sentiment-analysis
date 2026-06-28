from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# load TF-IDF and LR model at startup
ifidf = joblib.load("models/tfidf_vectorizer.pkl")
lr_model = joblib.load("models/lr_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    
    text_tran = ifidf.transform([text])
    prediction = lr_model.predict(text_tran)[0]
    proba = lr_model.predict_proba(text_tran)[0]
    label = "Positive" if prediction == 1 else "Negative"
    confidence = round(float(proba[prediction]) * 100, 2)
    return jsonify({"label": label, "confidence": confidence, "model": "tfidf"})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
