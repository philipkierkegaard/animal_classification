import os
import torch
import zipfile
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
from torchvision import transforms
from PIL import Image

# Initialize Kaggle API client
api = KaggleApi()
api.authenticate()

# Define the Kaggle dataset and directory
dataset_name = "antobenedetti/animals"  # Replace with your desired dataset
raw_data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")  # Path to 'data/raw'
processed_data_dir = os.path.join(os.path.dirname(__file__), "../../data/processed")  # Path to 'data/processed'

# Create the raw and processed data directories if they don't exist
os.makedirs(raw_data_dir, exist_ok=True)
os.makedirs(processed_data_dir, exist_ok=True)

# Define the file path where the dataset will be saved
dataset_path = os.path.join(raw_data_dir, "animals.zip")

# Function to download the dataset from Kaggle
def download_dataset(dataset_name, dest_path):
    print(f"Downloading dataset {dataset_name} from Kaggle...")
    api.dataset_download_files(dataset_name, path=dest_path, unzip=False)
    print(f"Dataset downloaded to {dest_path}")

# Function to extract the dataset (if needed)
def extract_dataset(file_path, dest_dir):
    print(f"Extracting dataset {file_path}...")
    
    # Check if the downloaded path is a directory (i.e., if it's a folder with a zip file inside)
    if os.path.isdir(file_path):
        nested_zip_path = os.path.join(file_path, "animals.zip")  # Path to the nested zip file
        if os.path.isfile(nested_zip_path):
            print(f"Found nested zip file: {nested_zip_path}")
            file_path = nested_zip_path  # Update to the nested zip file
        else:
            print(f"Error: No zip file found inside {file_path}.")
            return

    # Now, attempt to extract the file (whether it's directly a zip file or nested)
    if file_path.endswith(".zip"):
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            print(f"Dataset extracted to {dest_dir}")
        except zipfile.BadZipFile:
            print(f"Error: The file {file_path} is not a valid zip file.")
    elif file_path.endswith(".tar.gz") or file_path.endswith(".tgz"):
        try:
            with tarfile.open(file_path, 'r:gz') as tar_ref:
                tar_ref.extractall(dest_dir)
            print(f"Dataset extracted to {dest_dir}")
        except tarfile.TarError:
            print(f"Error: The file {file_path} is not a valid tar file.")
    else:
        print(f"Unsupported file format: {file_path}")

# Define the transformation pipeline
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

# Main execution
if __name__ == "__main__":
    # Download the dataset
    download_dataset(dataset_name, dataset_path)

    # Check if the file exists before trying to extract
    if os.path.exists(dataset_path):
        # Extract the dataset if it's a compressed file
        extract_dataset(dataset_path, raw_data_dir)
        
        # After extraction, process the images into tensors and save them
        print("Converting and saving images as tensors...")
        
        # Define paths for both train and validation sets
        train_image_dir = os.path.join(raw_data_dir, "animals/train")  # Path to the extracted images (train)
        val_image_dir = os.path.join(raw_data_dir, "animals/val")      # Path to the extracted images (val)
        
        # Create processed directories for both train and val
        processed_train_dir = os.path.join(processed_data_dir, "train")
        processed_val_dir = os.path.join(processed_data_dir, "val")
        
        # Convert and save training images as tensors
        convert_and_save_images(train_image_dir, processed_train_dir, transform)
        
        # Convert and save validation images as tensors
        convert_and_save_images(val_image_dir, processed_val_dir, transform)

    else:
        print(f"Error: The file {dataset_path} does not exist or is not a valid file.")

    # Path to the 'animals.zip' folder
    animals_zip_folder = os.path.join(raw_data_dir, "animals.zip")

    # Check if the directory exists before trying to delete it
    if os.path.isdir(animals_zip_folder):
        shutil.rmtree(animals_zip_folder)
        print(f"Deleted the folder: {animals_zip_folder}")
    else:
        print(f"The folder {animals_zip_folder} does not exist.")
