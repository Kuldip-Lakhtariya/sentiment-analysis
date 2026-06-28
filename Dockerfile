FROM python:3.11-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . . 

# download BERT model during build so it's ready at startup
RUN python -c "from transformers import BertTokenizer, BertForSequenceClassification; \
    BertTokenizer.from_pretrained('kuldip2611/sentiment-bert'); \
    BertForSequenceClassification.from_pretrained('kuldip2611/sentiment-bert')"

EXPOSE $PORT
CMD gunicorn --bind 0.0.0.0:$PORT app:app
