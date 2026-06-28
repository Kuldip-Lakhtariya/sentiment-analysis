from flask import Flask,request,jsonify,render_template
import joblib
import numpy as np
import torch 
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

ifidf = joblib.load("models/tfidf_vectorizer.pkl")
lr_model = joblib.load("models/lr_model.pkl")
bert_tokenizer = BertTokenizer.from_pretrained("models/bert_model")
bert_model = BertForSequenceClassification.from_pretrained("models/bert_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = bert_model.to(device)
bert_model.eval() 


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
        data = request.get_json()
        text = data["text"]
        model_choice = data["model"]
        print(text, model_choice) 

        if model_choice == "tfidf":
            text_tran = ifidf.transform([text])
            prediction = lr_model.predict(text_tran)[0]
            proba = lr_model.predict_proba(text_tran)[0]
            label = "Positive" if prediction == 1 else "Negative"
            confidence = round(float(proba[prediction]) * 100, 2)
            return jsonify({"label": label, "confidence": confidence, "model": "tfidf"})
        elif model_choice == "bert":
            inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output = bert_model(**inputs)
                preds = torch.argmax(output.logits, dim=1)
                probs = torch.softmax(output.logits, dim=1)[0]
                confidence = round(float(probs[preds].item()) * 100, 2)
                label = "Positive" if preds.item() == 1 else "Negative"
            return jsonify({"label": label, "confidence": confidence, "model": "bert"})
        
@app.route("/health",methods=["GET"])
def health():
        return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
