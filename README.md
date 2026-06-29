# IMDb Sentiment Analysis

An end-to-end NLP system that classifies movie reviews as positive or negative — built with two models (TF-IDF + Logistic Regression and fine-tuned BERT), deployed as a Flask REST API with a dark-themed interactive frontend.

**Live Demo:** [sentiment-analysis-1-8tkn.onrender.com](https://sentiment-analysis-1-8tkn.onrender.com)
*(Free tier — allow 30–60 seconds for cold start)*

**BERT Model (HuggingFace):** [kuldip2611/sentiment-bert](https://huggingface.co/kuldip2611/sentiment-bert)

---

## The Problem

Sentiment analysis is one of the most common real-world NLP tasks — product reviews, customer feedback, social media monitoring. The goal here was not just to build a classifier, but to compare a classical NLP approach against a modern transformer, understand where each wins and why, and ship both as a production API.

---

## Two Models, One API

Most projects build one model and deploy it. This project builds two with different architectures and exposes both through a single API — so the tradeoffs become visible in production, not just in a notebook.

| | TF-IDF + Logistic Regression | BERT (fine-tuned) |
|---|---|---|
| Accuracy | 88% | 87% |
| Training time | Seconds | ~45 min (Colab T4 GPU) |
| Model size | ~5MB | ~440MB |
| Inference speed | Instant | Slower |
| Deployed | Yes (Render) | HuggingFace Hub only |

BERT underperformed the baseline — not because transformers are worse, but because the fine-tuning setup wasn't optimal: 3 epochs with no warmup schedule on a dataset that the baseline already handles well. IMDb reviews are long, clean, and structured — exactly the kind of data where TF-IDF with good features is already a strong baseline. BERT's advantage shows more on short, ambiguous, or context-heavy text.

---

## Dataset

**Source:** IMDb Movie Reviews Dataset
**Size:** 50,000 reviews — perfectly balanced (25,000 positive / 25,000 negative)
**Split:** 80% training / 20% test

---

## Tech Stack

| Layer | Tool |
|---|---|
| Preprocessing | BeautifulSoup (HTML tag removal), scikit-learn |
| Baseline model | TF-IDF (max_features=10,000) + Logistic Regression |
| Transformer model | bert-base-uncased, fine-tuned via HuggingFace Transformers + PyTorch |
| Training (BERT) | Google Colab T4 GPU |
| API | Flask, dual-model endpoints |
| Server | Gunicorn |
| Container | Docker |
| Deployment | Render (TF-IDF), HuggingFace Hub (BERT) |

---

## Preprocessing Pipeline

Raw IMDb reviews contain HTML tags from web scraping. Cleaned with BeautifulSoup before any vectorization.

```
Raw review text
→ BeautifulSoup HTML tag removal
→ Lowercasing
→ TF-IDF vectorization (10,000 features)  ← for baseline
→ BERT tokenization (WordPiece, max_length=512)  ← for transformer
```

---

## Model Details

### Baseline — TF-IDF + Logistic Regression
- TF-IDF converts each review into a 10,000-dimensional sparse vector based on word frequency weighted by how unique that word is across all reviews
- Logistic Regression trained on these vectors
- Test accuracy: **88%**
- Fast, interpretable, deployable on minimal RAM

### BERT Fine-tuning
- Base model: `bert-base-uncased` (pretrained on BooksCorpus + Wikipedia)
- Fine-tuned on IMDb training set for 3 epochs on Colab T4 GPU
- PyTorch training loop with AdamW optimizer
- Test accuracy: **87%**
- Model size: 440MB — exceeds Render free tier RAM (512MB), so excluded from deployment
- Hosted on HuggingFace Hub: `kuldip2611/sentiment-bert`

Why BERT didn't beat the baseline here:
- 3 epochs is on the lower end — BERT typically benefits from a warmup scheduler that gradually increases learning rate before decaying it. This wasn't used.
- IMDb is a clean, well-structured dataset. Long-form reviews with clear vocabulary are exactly where TF-IDF performs strongly. BERT's contextual understanding matters more on short or ambiguous text.
- This is not a data problem or an architecture problem — it's a training configuration issue that would be fixed with a warmup schedule, more epochs, and learning rate tuning.

---

## API Endpoints

**POST /predict**
Runs TF-IDF + Logistic Regression (deployed model).

Request:
```json
{ "review": "The performances were outstanding and the story kept me engaged throughout." }
```

Response:
```json
{
  "sentiment": "positive",
  "confidence": 0.94,
  "model": "tfidf"
}
```

**POST /predict-bert**
Loads BERT from HuggingFace Hub (available when RAM permits).

**GET /health**
```json
{ "status": "running" }
```

---

## Frontend

Dark-themed single-page interface with:
- Text input for any movie review
- Real-time confidence bar showing prediction strength
- 4-level intensity system with contextual output (not just "positive/negative")
- Cycling example chips across 5 review categories — lets users try pre-written examples without typing

---

## Deployment Architecture

```
Flask app
  → Gunicorn (production WSGI server)
    → Docker container ($PORT via shell form CMD — required for Render)
      → Render free tier

BERT model (440MB) → HuggingFace Hub
TF-IDF model (~5MB) → included in Docker image
```

Why BERT is excluded from the Docker image: Render free tier provides ~512MB RAM. BERT at 440MB leaves only ~72MB for the rest of the process — not enough. The solution was to host BERT on HuggingFace Hub separately and keep the deployed API lean with only the TF-IDF model.

Why models are downloaded during Docker BUILD not runtime: Downloading at runtime causes Render startup timeout on the free tier. Build-time downloads are cached in the image layer.

---

## Repo Structure

```
sentiment-analysis/
│
├── notebook/
│   └── sentiment_analysis.ipynb   # Full EDA, training, evaluation
├── app.py                         # Flask API — dual model endpoints
├── tfidf_model.pkl                # TF-IDF vectorizer + LR model
├── templates/
│   └── index.html                 # Dark-themed frontend
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Run Locally

**Option 1 — Python**
```bash
git clone https://github.com/Kuldip-Lakhtariya/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
python app.py
```
Visit `http://localhost:5000`

**Option 2 — Docker**
```bash
docker build -t sentiment-api .
docker run -p 5000:5000 sentiment-api
```

---

## Key Learnings

- **A stronger model is not always a better deployment.** BERT is more capable than TF-IDF, but 440MB doesn't fit in 512MB of RAM. Architecture decisions in production depend on infrastructure constraints, not just accuracy numbers.
- **Baseline performance sets the bar — don't skip it.** Building TF-IDF first gave a concrete target (88%) and revealed that IMDb is a dataset where classical methods already perform well. Without the baseline, BERT's 87% would look like a success instead of a signal to investigate.
- **Fine-tuning BERT mirrors transfer learning in CNNs.** Freeze the pretrained weights, train only the classification head first, then optionally unfreeze. The same two-phase pattern from the plant disease project applies here — different domain, same principle.
- **Deployment constraints should be planned at the start, not discovered at the end.** Model size, RAM limits, and startup time all affect what can actually ship. These are design inputs, not afterthoughts.

---

## Planned Improvements

- [ ] Add warmup schedule and train BERT for more epochs — fix the training configuration issue
- [ ] Deploy BERT via a separate lightweight inference endpoint (HuggingFace Inference API)
- [ ] Add per-word attention visualization to show which words drove the prediction

---

## Author

**Kuldip Lakhtariya**
B.Tech ECE — LD College of Engineering, Ahmedabad
[GitHub](https://github.com/Kuldip-Lakhtariya) · [LinkedIn](https://www.linkedin.com/in/kuldip-lakhtariya-957106371/) · kuldip2611lakhtariya@gmail.com
