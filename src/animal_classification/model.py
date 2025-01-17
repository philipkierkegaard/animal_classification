import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class AnimalClassificationCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(AnimalClassificationCNN, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input channels = 3 (RGB)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Fully connected layers
        # Adjusted dimensions based on the image size after convolution and pooling
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjusted dimensions (128 channels, 16x16 spatial size)
        self.fc2 = nn.Linear(512, num_classes)    # Output layer (num_classes)

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)

        # Flatten the output of the last convolutional layer
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Example usage:
if __name__ == "__main__":
    # Instantiate the model
    model = AnimalClassificationCNN(num_classes=5)  # 5 classes: cat, dog, elephant, horse, lion
    
    # Print the model architecture
    print(model)
