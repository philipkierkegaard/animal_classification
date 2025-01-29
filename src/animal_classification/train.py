import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from model import AnimalClassificationCNN
from torch.profiler import profile, ProfilerActivity
from google.cloud import storage
import tempfile
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Custom Dataset to load tensors and labels from GCS bucket
class AnimalDataset(Dataset):
    def __init__(self, bucket_name, transform=None):
        self.bucket_name = bucket_name
        self.transform = transform
        self.labels = {'cat': 0, 'dog': 1, 'elephant': 2, 'horse': 3, 'lion': 4}  # Label map
        
        self.client = storage.Client()
        self.bucket = self.client.get_bucket(self.bucket_name)
        
        self.image_tensors = []
        self.image_labels = []

        for class_name, label in self.labels.items():
            class_tensor_path = f"data/processed/train/{class_name}_data.pt"
            blob = self.bucket.blob(class_tensor_path)
            
            if blob.exists():
                print(f"Loading tensor for class: {class_name} from {class_tensor_path}")
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    blob.download_to_filename(tmp_file.name)
                    class_tensor = torch.load(tmp_file.name)
                    self.image_tensors.append(class_tensor)
                    self.image_labels.extend([label] * class_tensor.shape[0])
            else:
                print(f"Error: Tensor file for class '{class_name}' not found at {class_tensor_path}")

        if not self.image_tensors:
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

# Load Model
model = AnimalClassificationCNN(num_classes=5)
model.load_state_dict(torch.load("models/animal_classification_model.pth", map_location=torch.device("cpu")))
model.eval()

# Define Image Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()
    
    labels = ["cat", "dog", "elephant", "horse", "lion"]
    return {"prediction": labels[prediction]}

# Training Function
def train(lr: float = 0.001, batch_size: int = 32, epochs: int = 1) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_bucket = "bucket_animal_classification"
    train_dataset = AnimalDataset(train_bucket, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "models/animal_classification_model.pth")

if __name__ == "__main__":
    import multiprocessing
    process = multiprocessing.Process(target=train)
    process.start()
    uvicorn.run(app, host="0.0.0.0", port=8080)
