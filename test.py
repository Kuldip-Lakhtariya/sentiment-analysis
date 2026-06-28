import requests

# test tfidf
r = requests.post("http://localhost:5000/predict", 
    json={"text": "This movie was amazing", "model": "tfidf"})
print("Status code:", r.status_code)
print("Raw response:", r.text)
print("TF-IDF:", r.json())

# test bert
r = requests.post("http://localhost:5000/predict",
    json={"text": "This movie was amazing", "model": "bert"})
print("Status code:", r.status_code)
print("Raw response:", r.text)
print("BERT:", r.json())