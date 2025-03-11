from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import numpy as np
import tensorflow as tf
import requests
from dotenv import load_dotenv
from pydantic import BaseModel, validator

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

load_dotenv()
API_KEY = os.getenv("API_KEY", "your-secret-api-key")
ASL_API_URL = os.getenv("ASL_API_URL", "/predict/")

# Serve static files with explicit path
app.mount("/static", StaticFiles(directory="frontend"), name="static")
# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://chandrakant06-asl-vision.hf.space",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Load model and class labels
model = tf.keras.models.load_model("backend/models/asl_landmark_model.h5")

CLASS_NAMES = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26,
    'nothing': 27, 'space': 28
}

# Reverse mapping
IDX_TO_CLASS = {v: k for k, v in CLASS_NAMES.items()}

class LandmarkRequest(BaseModel):
    landmarks: list

    @validator('landmarks')
    def validate_landmarks(cls, landmarks):
        # Validate landmarks input
        if not landmarks:
            raise ValueError("Landmarks cannot be empty")
        if len(landmarks) != 21:  # Assuming 21 landmark points
            raise ValueError(f"Expected 21 landmark points, got {len(landmarks)}")
        return landmarks

# Dependency to check API key
def verify_api_key(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Root endpoint to serve index.html
@app.get("/")
async def serve_index():
    return FileResponse("frontend/index.html")

@app.post("/predict/")
def predict(request: LandmarkRequest, api_key: str = Depends(verify_api_key)):
    try:
        # Reshape landmarks to match model input
        landmarks = np.array(request.landmarks).reshape(1, -1)
        
        # Predict
        predictions = model.predict(landmarks)[0]
        predicted_index = np.argmax(predictions)
        
        return {
            "prediction_label": int(predicted_index),
            "prediction_class": IDX_TO_CLASS[predicted_index],
            "confidence": float(predictions[predicted_index]),
            "all_confidences": {
                IDX_TO_CLASS[idx]: float(conf) 
                for idx, conf in enumerate(predictions)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Optional: Add error handling middleware
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail
    }
