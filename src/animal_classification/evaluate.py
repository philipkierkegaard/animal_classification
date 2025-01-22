import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
from model import AnimalClassificationCNN  # Import the model from model.py
from sklearn.metrics import accuracy_score
from google.cloud import storage
import tempfile

# Custom Dataset class to load tensor data for evaluation from GCS
class AnimalDataset(Dataset):
    def __init__(self, bucket_name, data_dir, transform=None):
        self.bucket_name = bucket_name
        self.data_dir = data_dir
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}  # Label map
        
        # Initialize Google Cloud Storage client
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        
        # Load all tensors (for each class)
        self.image_tensors = []
        self.image_labels = []
        
        for class_name, label in self.labels.items():
            class_tensor_path = os.path.join(data_dir, f"{class_name}_data.pt")
            blob = self.bucket.blob(class_tensor_path)
            
            if blob.exists():
                print(f"Downloading tensor for class: {class_name} from {class_tensor_path}")
                
                # Download tensor to a temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    blob.download_to_filename(temp_file.name)
                    class_tensor = torch.load(temp_file.name)
                
                self.image_tensors.append(class_tensor)
                self.image_labels.extend([label] * class_tensor.shape[0])  # Assign label to all images in this class
            else:
                print(f"Error: Tensor file for class '{class_name}' not found in {class_tensor_path}")

        if not self.image_tensors:
            print("No image tensors were loaded. Please check if the tensor files exist in the bucket.")
            raise ValueError("No tensors found in the processed data directory.")
        
        # Stack all class tensors into one large tensor
        self.image_tensors = torch.cat(self.image_tensors, dim=0)  # Concatenate along the first dimension (batch size)
        self.image_labels = torch.tensor(self.image_labels)  # Convert labels to tensor
    
    def __len__(self):
        return len(self.image_tensors)
    
    def __getitem__(self, idx):
        image = self.image_tensors[idx]
        label = self.image_labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for RGB
])

# Set bucket name and validation directory
bucket_name = "bucket_animal_classification"  # Replace with your GCS bucket name
val_dir = "data/processed/val"  # Path in GCS to validation data

# Set up dataset and data loader for validation
val_dataset = AnimalDataset(bucket_name, val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = AnimalClassificationCNN(num_classes=5)  # 5 classes: cat, dog, elephant, horse, lion

# Download the model from GCS
model_blob_path = "models/animal_classification_model.pth"
model_blob = storage.Client().bucket(bucket_name).blob(model_blob_path)

if model_blob.exists():
    print(f"Downloading model from GCS: {model_blob_path}")
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        model_blob.download_to_filename(temp_file.name)
        model.load_state_dict(torch.load(temp_file.name))
else:
    raise FileNotFoundError(f"Model file not found in GCS at {model_blob_path}")

# Set model to evaluation mode
model.eval()

# Initialize variables for accuracy calculation
all_preds = []
all_labels = []

# Evaluation loop
with torch.no_grad():  # No need to compute gradients for evaluation
    for inputs, labels in val_loader:
        # Forward pass
        outputs = model(inputs)
        
        # Get predictions
        _, preds = torch.max(outputs, 1)
        
        # Store predictions and true labels
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")
