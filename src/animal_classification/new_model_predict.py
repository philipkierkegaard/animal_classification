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

def predict_image(image, model, class_mapping):
    with torch.no_grad():
        outputs = model(image.unsqueeze(0))  # Add batch dimension
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_class = class_mapping[torch.argmax(probabilities).item()]
        class_probabilities = {class_mapping[i]: float(probabilities[i]) * 100 for i in range(len(class_mapping))}
        return predicted_class, class_probabilities

# Load the model from Google Cloud Storage
bucket_name = "bucket_animal_classification"  # Adjust the bucket name as needed
model_blob_path = "models/animal_classification_model.pth"  # Adjust the model path in GCS as needed
model = load_model(bucket_name, model_blob_path)

# Define the function Gradio will use
def gradio_interface(image):
    """
    Function for Gradio to handle an image input and return a prediction.
    """
    try:
        # Directly preprocess the PIL image
        processed_image = preprocess_image(image)

        # Predict using the loaded model
        predicted_class, class_probabilities = predict_image(processed_image, model, CLASS_MAPPING)

        # Format the output
        result = f"Predicted Class: {predicted_class}\n\nClass Probabilities:\n"
        result += "\n".join([f"{cls}: {prob:.2f}%" for cls, prob in class_probabilities.items()])

        return result
    except Exception as e:
        return f"Error during prediction: {str(e)}"

# Set up Gradio interface
gr.Interface(
    fn=gradio_interface,  # Your prediction function
    inputs=gr.Image(type="pil"),  # Image input
    outputs="text",  # Text output
    title="Animal Image Classification",
    description="Upload an animal image, and the model will classify it into one of the animal categories."
).launch(server_port=8080, server_name="0.0.0.0")
