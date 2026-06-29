from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel
from transformers import pipeline
import logging
import time
from typing import List # Import List for type hinting

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge # Import Gauge
from sklearn.metrics import precision_recall_fscore_support # Métriques de classification par catégorie

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Classifier API",
    description="API for classifying news articles into categories using a Hugging Face model.",
    version="1.5.0"
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

# Nouvelle métrique : Gauge pour l'accuracy du modèle
model_accuracy_score = Gauge(
    'model_accuracy_score',
    'Current accuracy of the News Classifier model',
    registry=registry
)

# Gauges precision / recall / f1 par catégorie
model_precision_score = Gauge(
    'model_precision_score',
    'Precision of the News Classifier model per category',
    ['category'],
    registry=registry
)

model_recall_score = Gauge(
    'model_recall_score',
    'Recall of the News Classifier model per category',
    ['category'],
    registry=registry
)

model_f1_score = Gauge(
    'model_f1_score',
    'F1 score of the News Classifier model per category',
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

        results = classifier(article.text)
        if not results:
            logger.error(f"Classifier returned empty results for text: {article.text[:50]}...")
            status_code = "500"
            raise HTTPException(status_code=500, detail="Model could not classify the text.")

        predicted_category = results[0]['label']
        confidence_score = results[0]['score']

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

    if not items:
        raise HTTPException(status_code=400, detail="No items provided for evaluation.")

    correct_predictions = 0
    total_predictions = len(items)

    y_true: List[str] = []
    y_pred: List[str] = []

    for item in items:
        try:
            prediction = classifier(item.text)[0]['label']
            true_label = item.true_label.upper() # Normaliser la casse pour la comparaison
            predicted = prediction.upper()
            y_true.append(true_label)
            y_pred.append(predicted)
            if predicted == true_label:
                correct_predictions += 1
        except Exception as e:
            logger.error(f"Error during evaluation for text: {item.text[:50]}... Error: {e}")

    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

    # Mettre à jour la métrique Prometheus Gauge (accuracy globale)
    model_accuracy_score.set(accuracy)

    # Calcul precision / recall / f1 par catégorie via scikit-learn
    if y_true:
        labels = sorted(set(y_true) | set(y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        for cat, p, r, f in zip(labels, precision, recall, f1):
            model_precision_score.labels(category=cat).set(p)
            model_recall_score.labels(category=cat).set(r)
            model_f1_score.labels(category=cat).set(f)

    logger.info(f"Model evaluated. Accuracy: {accuracy:.4f} on {total_predictions} items.")

    # Incrémenter les métriques d'API pour l'endpoint /evaluate
    api_requests_total.labels(endpoint="/evaluate", method="POST", status_code="200").inc()
    api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code="200").observe(time.time() - start_time)


    return {"message": "Model evaluation completed", "accuracy": accuracy, "evaluated_items": total_predictions}

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")
