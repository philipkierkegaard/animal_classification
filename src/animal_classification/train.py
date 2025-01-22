import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from model import AnimalClassificationCNN
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
from google.cloud import storage
import tempfile

# Custom Dataset to load tensors and labels from GCS bucket
class AnimalDataset(Dataset):
    def __init__(self, bucket_name, transform=None):
        self.bucket_name = bucket_name
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}  # Label map
        
        # Initialize Google Cloud Storage client
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)
        
        # List of tensors to load
        self.image_tensors = []
        self.image_labels = []

        # Load tensors for each class from GCS
        for class_name, label in self.labels.items():
            class_tensor_path = f"data/processed/train/{class_name}_data.pt"  # Path in your GCS bucket
            
            blob = self.bucket.blob(class_tensor_path)
            
            if blob.exists():
                print(f"Loading tensor for class: {class_name} from {class_tensor_path}")
                # Download the tensor file to a temporary location
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    blob.download_to_filename(tmp_file.name)
                    class_tensor = torch.load(tmp_file.name)
                    self.image_tensors.append(class_tensor)
                    self.image_labels.extend([label] * class_tensor.shape[0])
            else:
                print(f"Error: Tensor file for class '{class_name}' not found at {class_tensor_path}")

        if not self.image_tensors:
            print("No image tensors were loaded. Please check if the tensor files exist.")
            raise ValueError("No tensors found in the processed data directory.")
        
        self.image_tensors = torch.cat(self.image_tensors, dim=0)
        self.image_labels = torch.tensor(self.image_labels)

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        image = self.image_tensors[idx]
        label = self.image_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def train(lr: float = 0.001, batch_size: int = 2, epochs: int = 1) -> None:
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")
    run = wandb.init(
        project="Animals_mlops",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Initialize the dataset with the GCS bucket name
    train_bucket = "bucket_animal_classification"  # Replace with your bucket name
    train_dataset = AnimalDataset(train_bucket, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model, loss function, and optimizer
    model = AnimalClassificationCNN(num_classes=5).to(device)  # Move model to GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], 
        on_trace_ready=tensorboard_trace_handler("./log/"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        num_epochs = epochs
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            preds, targets = [], []

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                prof.step()

                accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
                wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})

                preds.append(outputs.detach().cpu())
                targets.append(labels.detach().cpu())

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"[{epoch+1}, {i+1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)

        print("Finished Training")
        
        final_accuracy = accuracy_score(targets, preds.argmax(dim=1))
        final_precision = precision_score(targets, preds.argmax(dim=1), average="weighted")
        final_recall = recall_score(targets, preds.argmax(dim=1), average="weighted")
        final_f1 = f1_score(targets, preds.argmax(dim=1), average="weighted")

        wandb.log({
            "final_accuracy": final_accuracy,
            "final_precision": final_precision,
            "final_recall": final_recall,
            "final_f1": final_f1
        })

        # Save the model locally
        local_model_path = "models/animal_classification_model.pth"
        os.makedirs("models", exist_ok=True)  # Ensure the directory exists
        torch.save(model.state_dict(), local_model_path)

        # Save the model to the GCS bucket
        gcs_model_path = "models/animal_classification_model.pth"  # Path in the bucket
        bucket_name = "bucket_animal_classification"  # Replace with your bucket name

        print(f"Uploading the model to GCS bucket {bucket_name} at {gcs_model_path}")
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(local_model_path)
        print("Model successfully uploaded to GCS!")

        # Log model artifact to W&B
        artifact = wandb.Artifact(
            name="Animals_mlops",
            type="model",
            description="A model trained to classify animal images",
            metadata={
                "accuracy": final_accuracy,
                "precision": final_precision,
                "recall": final_recall,
                "f1": final_f1,
            },
        )
        artifact.add_file(local_model_path)
        run.log_artifact(artifact)

if __name__ == "__main__":
    train()
