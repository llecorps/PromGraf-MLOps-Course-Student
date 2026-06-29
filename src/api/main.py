import logging
import time
import numpy as np

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
from transformers import pipeline
from typing import List
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classifier API",
    description="API for classifying news articles into categories using a Hugging Face model.",
    version="2.0.0"
)

registry = CollectorRegistry()

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

predictions_by_category = Counter(
    'predictions_by_category',
    'Number of predictions by category',
    ['category'],
    registry=registry
)

model_accuracy_score = Gauge(
    'model_accuracy_score',
    'Current accuracy of the News Classifier model',
    registry=registry
)

# Nouvelles métriques : Gauges pour Precision, Recall, F1-score par catégorie
model_precision_score = Gauge(
    'model_precision_score',
    'Precision score of the News Classifier model by category',
    ['category'],
    registry=registry
)

model_recall_score = Gauge(
    'model_recall_score',
    'Recall score of the News Classifier model by category',
    ['category'], # Avec un label pour la catégorie
    registry=registry
)

model_f1_score = Gauge(
    'model_f1_score',
    'F1 score of the News Classifier model by category',
    ['category'], # Avec un label pour la catégorie
    registry=registry
)

# Nouvel Histogramme pour la longueur des textes d'entrée
input_text_length_histogram = Histogram(
    'input_text_length_chars',
    'Length of input text in characters',
    registry=registry
)

# Nouvel Histogramme pour le score de confiance des prédictions
prediction_confidence_score_histogram = Histogram(
    'prediction_confidence_score',
    'Confidence score of model predictions',
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

# Modèle de données pour l'évaluation
class EvaluationItem(BaseModel):
    text: str
    true_label: str

@app.get("/")
async def read_root():
    return {"message": "Welcome to the News Classifier API. Use /predict to classify articles."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(article: ArticleInput):
    start_time = time.time()
    status_code = "200"

    try:
        if not article.text:
            logger.warning("Received empty text for prediction.")
            status_code = "400"
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Observer la longueur du texte d'entrée
        input_text_length_histogram.observe(len(article.text))
        results = classifier(article.text)
        if not results:
            logger.error(f"Classifier returned empty results for text: {article.text[:50]}...")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model could not classify the text.")

        predicted_category = results[0]['label']
        confidence_score = results[0]['score']

        predictions_by_category.labels(category=predicted_category).inc()
        # Observer le score de confiance
        prediction_confidence_score_histogram.observe(confidence_score)

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
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()

# Endpoint pour évaluer le modèle
@app.post("/evaluate")
async def evaluate_model(items: List[EvaluationItem]):
    """
    Evaluates the model on a given list of items with true labels.
    Updates the model_accuracy_score metric.
    """
    start_time = time.time()
    status_code = "200"

    try:
        if not items:
            status_code = "400"
            raise HTTPException(status_code=400, detail="No items provided for evaluation.")

        true_labels = []
        predicted_labels = []
        total_predictions = len(items)

        # Récupérer toutes les catégories uniques pour les métriques par catégorie
        all_categories = sorted(list(set([item.true_label.lower() for item in items] + [val.lower() for val in getattr(classifier, 'model').config.id2label.values()])))

        for item in items:
            try:
                prediction_result = classifier(item.text)
                predicted_label = prediction_result[0]['label']
                true_labels.append(item.true_label.lower())
                predicted_labels.append(predicted_label.lower())

            except Exception as e:
                logger.error(f"Error during single item evaluation for text: {item.text[:50]}... Error: {e}")
                total_predictions -= 1

        if not true_labels:
            raise HTTPException(status_code=500, detail="No successful predictions during evaluation.")

        # Calcul de l'Accuracy globale
        accuracy = accuracy_score(true_labels, predicted_labels)
        model_accuracy_score.set(accuracy)

        # Calcul de Precision, Recall, F1-score par catégorie
        p, r, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, labels=all_categories, average=None, zero_division=0)

        for i, category in enumerate(all_categories):
            model_precision_score.labels(category=category).set(p[i])
            model_recall_score.labels(category=category).set(r[i])
            model_f1_score.labels(category=category).set(f1[i])

        logger.info(f"Model evaluated. Accuracy: {accuracy:.4f} on {total_predictions} items. Per-category scores updated.")
        return {"message": "Model evaluation completed", "accuracy": accuracy, "evaluated_items": total_predictions}

    except HTTPException as e:
        status_code = str(e.status_code)
        raise
    except Exception as e:
        logger.error(f"Error during evaluation of model: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Model evaluation failed due to an internal error: {e}")
    finally:
        end_time = time.time()
        duration = end_time - start_time
        api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code=status_code).observe(duration)
        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code=status_code).inc()

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")
