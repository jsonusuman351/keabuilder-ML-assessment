"""
KeaBuilder ML Microservice
- Demonstrates how to serve ML model from Node.js backend
- FastAPI service that Node.js calls via REST
- Author: Suman
"""

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI(title="KeaBuilder ML Service")

class PredictionRequest(BaseModel):
    user_id: str
    input_text: str
    input_type: str = "chat"

class PredictionResponse(BaseModel):
    input_id: str
    prediction: dict
    confidence: float
    model_version: str

def lead_score_model(text: str) -> tuple:
    """Mock ML model - replace with real model in production"""
    keywords = ['buy', 'price', 'demo', 'checkout', 'subscribe', 'plan']
    score = sum(1 for k in keywords if k in text.lower()) / len(keywords)
    label = "hot" if score > 0.3 else "warm" if score > 0.1 else "cold"
    return {"score": round(score, 3), "label": label}, round(min(score + 0.5, 1.0), 3)

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    prediction, confidence = lead_score_model(req.input_text)
    return PredictionResponse(
        input_id=f"inp_{np.random.randint(10000)}",
        prediction=prediction,
        confidence=confidence,
        model_version="1.0.0"
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "keabuilder-ml"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
