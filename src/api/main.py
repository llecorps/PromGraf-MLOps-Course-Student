from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
from transformers import pipeline
import logging
import os
import time # time pour mesurer les latences

# Prometheus metrics types
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classifier API",
    description="API for classifying news articles into categories using a Hugging Face model.",
    version="1.1.0"
)

# --- Définition des metrics ---
registry = CollectorRegistry()

# Counter 'api_requests_total', label par endpoint, method, et status code
api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

# Histogram 'api_request_duration_seconds', label par endpoint, method, et status code
api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

# Counter 'predictions_by_category'
predictions_by_category = Counter(
    'predictions_by_category',
    'Number of predictions by category',
    ['category'],
    registry=registry
)

try:
    classifier = pipeline("text-classification", model="dima806/news-category-classifier-distilbert")
    logger.info("Hugging Face model loaded successfully: dima806/news-category-classifier-distilbert")
except Exception as e:
    logger.error(f"Error loading Hugging Face model: {e}")
    raise RuntimeError("Failed to load ML model, application cannot start.") from e

class ArticleInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    category: str
    score: float

@app.get("/")
async def read_root():
    return {"message": "Welcome to the News Classifier API. Use /predict to classify articles."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(article: ArticleInput):
    """
    Classify a news article based on its text content.
    """
    start_time = time.time() # Début du timer pour la durée de la requête

    status_code = "200"

    try:
        if not article.text:
            logger.warning("Received empty text for prediction.")
            status_code = "400"
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        results = classifier(article.text)
        if not results:
            logger.error(f"Classifier returned empty results for text: {article.text[:50]}...")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model could not classify the text.")

        predicted_category = results[0]['label']
        confidence_score = results[0]['score']

        # Incrementation du counter pour la catégorie prédite
        predictions_by_category.labels(category=predicted_category).inc()

        logger.info(f"Classified text: '{article.text[:50]}...' into category: '{predicted_category}' with score: {confidence_score:.4f}")
        return PredictionOutput(category=predicted_category, score=confidence_score)

    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during prediction for text: {article.text[:50]}... Error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed due to an internal error: {e}")
    finally:
        end_time = time.time()
        # Durée de la requête
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")
