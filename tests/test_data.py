import os
import torch
import pytest
from src.animal_classification.model import AnimalClassificationCNN

"""
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

"""

def test_transform():
    """Test the transform pipeline."""
    from src.animal_classification.data import transform
    from PIL import Image
    import numpy as np

    # Create a dummy image (RGB)
    dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255), "RGB")
    
    # Apply the transform
    transformed = transform(dummy_image)
    
    # Assertions
    assert isinstance(transformed, torch.Tensor), "Transformed image should be a tensor"
    assert transformed.shape == (3, 128, 128), "Transformed tensor should have shape (3, 128, 128)"

