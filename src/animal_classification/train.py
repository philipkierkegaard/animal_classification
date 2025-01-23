import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import AnimalClassificationCNN
from google.cloud import storage
import tempfile
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import argparse
import yaml
from google.api_core.exceptions import GoogleAPICallError, NotFound
from google.api_core.retry import Retry


# Custom function to download GCS blobs with retry
def download_blob_with_retry(blob, filename):
    retry = Retry(deadline=30)  # Retry for up to 30 seconds
    try:
        blob.download_to_filename(filename, retry=retry)
    except (GoogleAPICallError, NotFound) as e:
        print(f"Failed to download {blob.name}: {e}")


# Custom Dataset to load tensors and labels from GCS bucket
class AnimalDataset(Dataset):
    def __init__(self, bucket_name, data_path, transform=None):
        self.bucket_name = bucket_name
        self.data_path = data_path
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}

        # Initialize GCS client
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)

        self.image_tensors = []
        self.image_labels = []

        for class_name, label in self.labels.items():
            class_tensor_path = f"{self.data_path}/{class_name}_data.pt"
            blob = self.bucket.blob(class_tensor_path)

            if blob.exists():
                print(f"Downloading tensor for class: {class_name} from {class_tensor_path}")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    download_blob_with_retry(blob, tmp_file.name)
                    class_tensor = torch.load(tmp_file.name)
                    self.image_tensors.append(class_tensor)
                    self.image_labels.extend([label] * class_tensor.shape[0])
            else:
                print(f"Tensor file for class '{class_name}' not found at {class_tensor_path}")

        if not self.image_tensors:
            raise ValueError("No tensors found in the dataset directory.")

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


def train(lr=0.001, batch_size=32, epochs=10, val_split=0.2):
    print(f"Training with lr={lr}, batch_size={batch_size}, epochs={epochs}")

    # Initialize W&B
    run = wandb.init(project="Animals_mlops", config={"lr": lr, "batch_size": batch_size, "epochs": epochs})

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define transformation
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    bucket_name = "bucket_animal_classification"
    data_path = "data/processed/train"
    full_dataset = AnimalDataset(bucket_name, data_path, transform=transform)
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss
    model = AnimalClassificationCNN(num_classes=5).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Profiling
    with profile(
        activities=[ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else []),
        on_trace_ready=tensorboard_trace_handler("./log/"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:

        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

            val_accuracy = val_correct / val_total
            print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {val_accuracy:.4f}")
            wandb.log({"val_loss": val_loss / len(val_loader), "val_accuracy": val_accuracy})

        # Profiling step
        prof.step()

    # Save the trained model
    model_path = "models/animal_classification_model.pth"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Upload model to GCS
    bucket = storage.Client().bucket(bucket_name)
    blob = bucket.blob(model_path)
    blob.upload_from_filename(model_path)
    print(f"Model uploaded to GCS bucket: {bucket_name}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run W&B sweep")
    args = parser.parse_args()

    if args.sweep:
        sweep_train()
    else:
        train()
