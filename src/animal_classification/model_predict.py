import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import AnimalClassificationCNN  # Import your model
from google.cloud import storage
import tempfile
import matplotlib.pyplot as plt

# Define class labels
labels = {0: 'cat', 1: 'dog', 2: 'elephant', 3: 'horse', 4: 'lion'}

# Initialize and load the model
bucket_name = "bucket_animal_classification"
model_blob_path = "models/animal_classification_model.pth"
model = AnimalClassificationCNN(num_classes=len(labels))

client = storage.Client()
model_blob = client.bucket(bucket_name).blob(model_blob_path)

if model_blob.exists():
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        model_blob.download_to_filename(temp_file.name)
        state_dict = torch.load(temp_file.name, map_location=torch.device('cpu'))
        print(f"Loaded state dict keys: {list(state_dict.keys())}")
        try:
            model.load_state_dict(state_dict)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
else:
    raise FileNotFoundError("Model weights not found.")

model.eval()

# Function to visualize preprocessing
def visualize_preprocessing(image, preprocessed_image):
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")

    inv_normalize = transforms.Normalize(
        mean=[-m/s for m, s in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1/s for s in [0.229, 0.224, 0.225]]
    )
    denorm_image = inv_normalize(preprocessed_image.squeeze())
    plt.subplot(1, 2, 2)
    plt.imshow(denorm_image.permute(1, 2, 0).numpy())
    plt.title("Preprocessed Image")
    plt.axis("off")
    plt.show()

# Prediction function
def predict(image):
    try:
        if not isinstance(image, Image.Image):
            return "Error: Please upload a valid image."

        # Define preprocessing
        preprocess = transforms.Compose([
            transforms.Resize((128, 128)),  # Match the size used during training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        preprocessed_image = preprocess(image).unsqueeze(0)  # Add batch dimension

        # Visualize preprocessing
        visualize_preprocessing(image, preprocessed_image[0])

        # Perform prediction
        with torch.no_grad():
            outputs = model(preprocessed_image)

            # Debugging: Check the raw logits
            print(f"Raw logits: {outputs}")

            probabilities = torch.softmax(outputs, dim=1)

            # Debugging: Check the softmax probabilities
            print(f"Softmax probabilities: {probabilities}")

            predictions = {labels[i]: float(probabilities[0, i]) for i in range(len(labels))}
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

            # Debugging: Check the final predictions
            print(f"Sorted predictions: {sorted_preds}")

            # Return formatted results
            return "\n".join([f"{label}: {confidence:.2%}" for label, confidence in sorted_preds])

    except Exception as e:
        return f"Error: {str(e)}"

# Test the model with a dummy input
def test_model():
    print("Testing model with dummy input...")
    dummy_input = torch.randn(1, 3, 128, 128)  # Batch size of 1, RGB image, size 128x128
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"Test output logits: {test_output}")
        print(f"Softmax probabilities: {torch.softmax(test_output, dim=1)}")

# Run the test to verify the model works correctly
test_model()

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Animal Classifier",
    description="Upload an animal image to classify it (e.g., cat, dog, elephant, horse, lion).",
)

if __name__ == "__main__":
    interface.launch(share=False)
