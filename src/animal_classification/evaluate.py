import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import transforms
from model import AnimalClassificationCNN  # Import the model from model.py
from sklearn.metrics import accuracy_score

# Custom Dataset class to load tensor data for evaluation
class AnimalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}  # Label map
        
        # Load all tensors (for each class)
        self.image_tensors = []
        self.image_labels = []
        
        for class_name, label in self.labels.items():
            class_tensor_path = os.path.join(data_dir, f"{class_name}_data.pt")
            
            # Check if the tensor file exists
            if os.path.exists(class_tensor_path):
                print(f"Loading tensor for class: {class_name} from {class_tensor_path}")
                class_tensor = torch.load(class_tensor_path)  # Load the tensor of images
                self.image_tensors.append(class_tensor)
                self.image_labels.extend([label] * class_tensor.shape[0])  # Assign label to all images in this class
            else:
                print(f"Error: Tensor file for class '{class_name}' not found at {class_tensor_path}")

        # If no tensors were loaded, print a message and raise an error
        if not self.image_tensors:
            print("No image tensors were loaded. Please check if the tensor files exist.")
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

# Set up dataset and data loader for validation
val_dir = "data/processed/val"  # Path to processed validation data

val_dataset = AnimalDataset(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize the model
model = AnimalClassificationCNN(num_classes=5)  # 5 classes: cat, dog, elephant, horse, lion
model.load_state_dict(torch.load("models/animal_classification_model.pth"))  # Load the trained model

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
