import os
import torch
import pytest
from src.animal_classification.model import AnimalClassificationCNN

# Function to test that processed .pt files exist for each class (animal)
def test_processed_data_files():
    data_dir = "data/processed/train"  # Path to the processed data directory
    
    # List the expected classes (these should match the tensor files you expect)
    expected_classes = ['cat', 'dog', 'elephant', 'horse', 'lion']
    
    # Check that a .pt file exists for each class
    for class_name in expected_classes:
        tensor_file_path = os.path.join(data_dir, f"{class_name}_data.pt")
        
        # Assert the file exists
        assert os.path.exists(tensor_file_path), f"Tensor file for class '{class_name}' not found at {tensor_file_path}"
        
        # Load the tensor to ensure it contains data
        tensor = torch.load(tensor_file_path, weights_only=True)
        assert tensor.shape[0] > 0, f"The tensor for class '{class_name}' is empty."
        
        # Assert the tensor is of the expected type (should be a 4D tensor: batch_size, channels, height, width)
        assert len(tensor.shape) == 4, f"The tensor for class '{class_name}' should have 4 dimensions"
        assert tensor.shape[1] == 3, f"The tensor for class '{class_name}' should have 3 channels (RGB)"
        assert tensor.shape[2] == 128 and tensor.shape[3] == 128, f"The tensor for class '{class_name}' should have shape (128, 128)"




# Test if the model's output shape matches the expected shape
def test_model_output_shape():
    num_classes = 5  # Assuming 5 classes: cat, dog, elephant, horse, lion
    model = AnimalClassificationCNN(num_classes=num_classes)

    # Create a random input tensor of the shape (batch_size, channels, height, width)
    # For example, a batch of 32 images, each with 3 channels (RGB), 128x128 pixels
    input_tensor = torch.randn(32, 3, 128, 128)  # Batch size of 32

    # Perform a forward pass through the model
    output_tensor = model(input_tensor)

    # Assert the output tensor shape is (batch_size, num_classes)
    assert output_tensor.shape == (32, num_classes), \
        f"Expected output shape (32, {num_classes}), but got {output_tensor.shape}"