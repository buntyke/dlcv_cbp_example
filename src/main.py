# import modules
import os
import sys
import logging
from pathlib import Path

# import third party imports
import uvicorn
from fastapi import FastAPI, File, UploadFile

# relative imports
from inference import load_model, preprocess_image, predict_image

# instantiate fastapi app
app = FastAPI()

# setup logger for debugging
logging.basicConfig(level=logging.INFO)

# load model and class names
artifact_path = 'models/'
model_path = os.path.join(artifact_path, 'resnet50.pth')
labels_path = os.path.join(artifact_path, 'imagenet_classes.txt')
model, model_classes = load_model(model_path, labels_path)

logging.info("Model loaded successfully")

@app.get("/health")
def health_check():
    """
    Health check endpoint
    
    Returns:
        Dict: Message indicating the API is up and running
    """
    return {"message": "Image Classifier API is up and running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Prediction endpoint

    Args:
        file (UploadFile): Image file to classify

    Returns:
        Dict: Predicted class
    """
    image = preprocess_image(await file.read())
    logging.info(f"Image tensor shape: {image.shape}")

    results = predict_image(model, model_classes, image)

    return {"prediction": results}

if __name__ == "__main__":
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
