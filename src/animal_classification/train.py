import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from model import AnimalClassificationCNN
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, precision_score, recall_score
import numpy as np

# Custom Dataset to load tensors and labels
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

def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 1) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    run = wandb.init(
        project="Animals_mlops",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize for RGB
    ])

    # Set up dataset and data loader
    train_dir = "data/processed/train"  # Path to processed data

    train_dataset = AnimalDataset(train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = AnimalClassificationCNN(num_classes=5)  # 5 classes: cat, dog, elephant, horse, lion
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop

    num_epochs = epochs  # Number of epochs to train

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        preds, targets = [], []
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            
            preds.append(outputs.detach().cpu())
            targets.append(labels.detach().cpu())
            
            running_loss += loss.item()  # Accumulate loss
            
            if i % 100 == 99:  # Print loss every 100 batches
                print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    # Concatenate the predictions and targets to evaluate final metrics
    preds = torch.cat(preds, 0)  # Shape: (num_samples, num_classes)
    targets = torch.cat(targets, 0)  # Shape: (num_samples,)

    print("Finished Training")
    
    # Calculate final metrics after training
    final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
    final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
    final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
    final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

    # Log final metrics to WandB
    wandb.log({
        "final_accuracy": final_accuracy,
        "final_precision": final_precision,
        "final_recall": final_recall,
        "final_f1": final_f1
    })

    # Save the trained model
    torch.save(model.state_dict(), "models/animal_classification_model.pth")
    artifact = wandb.Artifact(
        name="Animals_mlops",
        type="model",
        description="A model trained to classify animals images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/animal_classification_model.pth")
    run.log_artifact(artifact)

if __name__ == "__main__":
    train()