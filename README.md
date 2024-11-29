# Image Classifier API

## Introduction
This project demonstrates an API for an image classification model using the ResNet-50 architecture trained on the ImageNet dataset. The API provides two endpoints:
1. `/health` - A health check endpoint to verify that the classifier is operational.
2. `/predict` - An endpoint to upload an image and receive the top 3 most probable classes along with their confidence scores.

The API makes it easy to classify images and assess the confidence of the predictions.

---

## Model Details
- Model Architecture: ResNet-50
- Dataset: ImageNet
- Input Requirements:
  - Image size: 224x224 pixels
  - Pixel normalization: Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225]
- Output: Logit values for 1000 classes in the ImageNet dataset, from which probabilities are derived using softmax.

---

## App Architecture
The application uses the FastAPI framework to expose the endpoints. Key components of the app include:
- Endpoints:
  - `/health` - Returns a status response indicating the API is operational.
  - `/predict` - Accepts an image file (PNG or JPEG), preprocesses it, and returns the top 3 predictions with confidence scores.
- Backend Framework: FastAPI
- Model Serving: PyTorch
- Deployment: Uvicorn is used as the ASGI server.

---

## Installation Instructions

### Prerequisites
1. Install Visual Studio Code for development.
2. Install Anaconda to manage Python virtual environments.
3. Ensure Python version 3.8 or higher is installed.

### Steps
1. Clone the repository and navigate to its directory.
2. Create and activate a virtual environment using Anaconda.
3. Install the required libraries from the `requirements.txt` file.
4. Run the `src/download.py` script to download the model artifacts.

---

## Usage Instructions
1. Start the API using Uvicorn.
2. Open a web browser (e.g., Chrome) and navigate to the localhost URL.
3. Use the endpoints:
   - Health Check: Verify the classifier is operational by accessing the `/health` endpoint.
   - Prediction: Upload an image (PNG or JPEG) via the `/predict` endpoint to receive classification results.

---

## Troubleshooting Instructions
1. Enable debug logging by setting the logging level to debug to get additional information.
2. Ensure the `load_model` function includes the `model.eval()` line for proper inference.
3. Verify the repository root directory is included in the system path before running Uvicorn.
4. Check that the uploaded image is in a supported format (PNG or JPEG) and meets the input requirements.

---

Happy classifying! If you encounter any issues, please refer to the troubleshooting section or open an issue in the repository.