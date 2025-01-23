import torch
from google.cloud import storage
import tempfile
from model import AnimalClassificationCNN
from PIL import Image
import gradio as gr
from torchvision import transforms

# Define class mapping
CLASS_MAPPING = {0: 'cat', 1: 'dog', 2: 'elephant', 3: 'horse', 4: 'lion'}

def download_model_from_gcs(bucket_name, model_blob_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(model_blob_path)

    if not blob.exists():
        raise FileNotFoundError(f"Model weights not found at gs://{bucket_name}/{model_blob_path}")

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        blob.download_to_filename(temp_file.name)
        return temp_file.name

def load_model(bucket_name, model_blob_path):
    model_file_path = download_model_from_gcs(bucket_name, model_blob_path)
    model = AnimalClassificationCNN(num_classes=len(CLASS_MAPPING))
    model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def predict(image):
    model = load_model("bucket_animal_classification", "models/animal_classification_model.pth")
    image = preprocess_image(image)
    image = image.unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    return CLASS_MAPPING[predicted.item()]

# Define Gradio interface
iface = gr.Interface(fn=predict, inputs="image", outputs="text")

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)