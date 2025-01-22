from fastapi import FastAPI, UploadFile, HTTPException
from torchvision import transforms
from google.cloud import storage
import torch
from PIL import Image
import io
import os

# Initialize FastAPI
app = FastAPI()

# Class names
class_names = ["cat", "dog", "elephant", "horse", "lion"]

# Transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Cloud Storage configuration
BUCKET_NAME = "bucket_animal_classification"
MODEL_PATH = "models/animal_classification_model.pth"

def download_model():
    """Download the model from Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(MODEL_PATH)
    
    # Ensure local directory exists
    os.makedirs("models", exist_ok=True)
    local_path = "models/animal_classification_model.pth"
    
    if not os.path.exists(local_path):
        print("Downloading model from GCS...")
        blob.download_to_filename(local_path)
        print("Model downloaded successfully.")
    else:
        print("Model already exists locally.")
    return local_path

# Load the model
model_path = download_model()
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        # Read and preprocess the image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)

        # Run the model
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted_idx = torch.max(outputs, 1)

        # Get the predicted class
        predicted_class = class_names[predicted_idx.item()]
        return {"class": predicted_class}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
