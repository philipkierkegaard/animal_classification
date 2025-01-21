import torch
from src.animal_classification.model import AnimalClassificationCNN


# Test to check the model's output shape matches expectations
def test_model_output_shape():
    model = AnimalClassificationCNN(num_classes=5)
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 128, 128)  # Input shape: (batch_size, channels, height, width)

    output = model(input_tensor)  # Forward pass
    assert output.shape == (batch_size, 5), f"Expected output shape (4, 5), but got {output.shape}"



# Test to check if the model can handle very small batches
def test_model_with_single_sample():
    model = AnimalClassificationCNN(num_classes=5)
    single_input = torch.randn(1, 3, 128, 128)  # Batch size of 1

    output = model(single_input)
    assert output.shape == (1, 5), f"Expected output shape (1, 5), but got {output.shape}"


# Test to ensure all model parameters are trainable
def test_model_parameters():
    model = AnimalClassificationCNN(num_classes=5)
    all_trainable = all(param.requires_grad for param in model.parameters())
    assert all_trainable, "Not all model parameters are set to require gradients"


# Test the forward pass functionality with a dummy input
def test_model_forward_pass():
    model = AnimalClassificationCNN(num_classes=5)
    dummy_input = torch.randn(4, 3, 128, 128)  # Batch size of 4
    try:
        _ = model(dummy_input)
    except Exception as e:
        assert False, f"Model forward pass failed with error: {str(e)}"


# Test dropout layer functionality
def test_model_dropout_effect():
    model = AnimalClassificationCNN(num_classes=5)
    model.eval()  # Set to evaluation mode to disable dropout
    dummy_input = torch.randn(1, 3, 128, 128)
    output_no_dropout = model(dummy_input)

    model.train()  # Set to training mode to enable dropout
    output_with_dropout = model(dummy_input)

    assert not torch.equal(output_no_dropout, output_with_dropout), \
        "Dropout does not seem to be applied correctly during training"
