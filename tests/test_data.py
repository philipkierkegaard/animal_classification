import pytest
import torch
from src.animal_classification.data import (
    download_and_move_dataset,
    convert_and_save_images,
    transform,
)

def test_transform():
    """Test the transform pipeline."""
    from PIL import Image
    import numpy as np

    # Create a dummy image (RGB)
    dummy_image = Image.fromarray(np.uint8(np.random.rand(256, 256, 3) * 255), "RGB")
    
    # Apply the transform
    transformed = transform(dummy_image)
    
    # Assertions
    assert isinstance(transformed, torch.Tensor), "Transformed image should be a tensor"
    assert transformed.shape == (3, 128, 128), "Transformed tensor should have shape (3, 128, 128)"
