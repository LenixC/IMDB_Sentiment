from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import pipeline
from prometheus_client import Counter, Histogram, start_http_server
import time

# Start prometheus metrics server on port 8001
start_http_server(8001)

app = FastAPI()

# Load the trained model and tokenizer from the local directory
model_path = "./model"  # Path to your saved model
tokenizer = AutoTokenizer.from_pretrained(model_path)
trained_model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create pipelines
naive_classifier = pipeline("sentiment-analysis", device=-1)  # CPU only for naive model
trained_classifier = pipeline("sentiment-analysis", model=trained_model, tokenizer=tokenizer, device=-1)  # CPU only for trained model

# Metrics
PREDICTION_TIME = Histogram('prediction_duration_seconds', 'Time spent processing prediction')
REQUESTS = Counter('prediction_requests_total', 'Total requests')
SENTIMENT_SCORE = Histogram('sentiment_score', 'Histogram of sentiment scores', buckets=[0.0, 0.25, 0.5, 0.75, 1.0])

class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    text: str
    sentiment: str
    score: float

@app.post("/predict/naive", response_model=SentimentOutput)
async def predict_naive_sentiment(input_data: TextInput):
    REQUESTS.inc()
    start_time = time.time()
    
    result = naive_classifier(input_data.text)[0]
    
    score = result["score"]
    SENTIMENT_SCORE.observe(score)  # Record the sentiment score
    
    PREDICTION_TIME.observe(time.time() - start_time)
    
    return SentimentOutput(
        text=input_data.text,
        sentiment=result["label"],
        score=score
    )

@app.post("/predict/trained", response_model=SentimentOutput)
async def predict_trained_sentiment(input_data: TextInput):
    REQUESTS.inc()
    start_time = time.time()
    
    result = trained_classifier(input_data.text)[0]
    
    score = result["score"]
    SENTIMENT_SCORE.observe(score)  # Record the sentiment score
    
    PREDICTION_TIME.observe(time.time() - start_time)
    
    return SentimentOutput(
        text=input_data.text,
        sentiment=result["label"],
        score=score
    )
