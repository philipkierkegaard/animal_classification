import os
import shutil
import torch
import torchvision.transforms as transforms
from PIL import Image
import kagglehub
from pathlib import Path

def download_and_move_dataset():
    # Specify the dataset name from Kaggle
    dataset_name = "antobenedetti/animals"
    
    # Specify the directory paths
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    
    # Ensure both directories exist; create them if they don't
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)

    # Download the dataset
    print("Downloading dataset...")
    path = kagglehub.dataset_download(dataset_name)
    
    print("Dataset downloaded successfully!")
    print("Dataset files are located at:", path)

    # Move the downloaded dataset to 'data/raw'
    print("Moving dataset to 'data/raw'...")
    dataset_name = os.path.basename(path)
    target_path = os.path.join(raw_data_dir, dataset_name)

    # Move the dataset folder (or file) to the target directory
    shutil.move(path, target_path)
    print(f"Dataset moved to: {target_path}")
    
    return target_path

transform = transforms.Compose([
    transforms.Resize((128, 128)),   # Resize image to 128x128 (you can adjust this)
    transforms.ToTensor(),           # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for RGB
])

# Function to convert and save images as tensors
def convert_and_save_images(image_dir, dest_dir, transform):
    class_folders = [f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))]
    print(f"Found {len(class_folders)} class folders: {class_folders}")
    
    # Check if the destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created processed data directory at {dest_dir}")
    
    for label, class_folder in enumerate(class_folders):
        class_folder_path = os.path.join(image_dir, class_folder)
        print(f"Processing class: {class_folder}")

        image_files = [f for f in os.listdir(class_folder_path) if f.endswith('.jpg')]
        images = []
        for image_file in image_files:
            image_path = os.path.join(class_folder_path, image_file)
            img = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
            img_tensor = transform(img)
            images.append(img_tensor)

        # Stack all images in this class into one tensor
        class_tensor = torch.stack(images, dim=0)
        
        # Save the tensor to the processed directory with the class label as part of the filename
        tensor_save_path = os.path.join(dest_dir, f"{class_folder}_data.pt")
        torch.save(class_tensor, tensor_save_path)
        print(f"Saved tensor for {class_folder} at {tensor_save_path}")

def main():
    # Step 1: Download and move the dataset
    data_dir = download_and_move_dataset()
    print("Dataset available at:", data_dir)

    # Step 2: Process images, convert to tensors, and move to 'data/processed'
    # Define paths for both train and validation sets
    train_image_dir = os.path.join(data_dir, "animals", "train")  # Corrected path to the train images
    val_image_dir = os.path.join(data_dir, "animals", "val")      # Corrected path to the validation images
    
    # Create processed directories for both train and val
    processed_train_dir = os.path.join("data/processed", "train")
    processed_val_dir = os.path.join("data/processed", "val")
    
    # Convert and save training images as tensors
    print("Processing training images...")
    convert_and_save_images(train_image_dir, processed_train_dir, transform)
    
    # Convert and save validation images as tensors
    print("Processing validation images...")
    convert_and_save_images(val_image_dir, processed_val_dir, transform)

    print("Finished processing all images.")

if __name__ == "__main__":
    main()
