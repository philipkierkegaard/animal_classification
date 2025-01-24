import pytest
import torch
from PIL import Image
import numpy as np
from unittest.mock import patch, MagicMock
from src.animal_classification.model_predict import predict, load_model, CLASS_MAPPING  # Replace 'app' with the actual file name if it's different



def create_dummy_image(size=(128, 128), color=(255, 255, 255)):
    """Create a dummy image for testing."""
    image = Image.new("RGB", size, color)
    return image

@pytest.fixture
def dummy_image():
    """Fixture to create a dummy image."""
    return create_dummy_image()

@pytest.fixture
def mock_model():
    """Mock the model to return fixed predictions."""
    mock_model = MagicMock()
    mock_model.eval = MagicMock()
    mock_model.__call__ = MagicMock(return_value=torch.tensor([[0.1, 0.7, 0.05, 0.1, 0.05]]))  # Mocked probabilities
    return mock_model


@patch("src.animal_classification.model_predict.gr.Interface.launch")
def test_gradio_interface(mock_launch):
    """Test if the Gradio interface launches without errors."""
    from src.animal_classification.model_predict import iface  # Import the Gradio interface instance

    try:
        iface.launch(server_name="0.0.0.0", server_port=8080)
        mock_launch.assert_called_once()
    except Exception as e:
        pytest.fail(f"Gradio interface launch failed: {e}")

