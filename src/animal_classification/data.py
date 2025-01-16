import os
import shutil
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile
import tarfile

# Initialize Kaggle API client
api = KaggleApi()
api.authenticate()

# Define the Kaggle dataset and directory
dataset_name = "antobenedetti/animals"  # Replace with your desired dataset
raw_data_dir = os.path.join(os.path.dirname(__file__), "../../data/raw")  # Path to 'data/raw'

# Create the raw data directory if it doesn't exist
os.makedirs(raw_data_dir, exist_ok=True)

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

    # Clean up by removing the unnecessary "animals.zip" folder
    if os.path.isdir(os.path.join(dest_dir, "animals.zip")):
        print(f"Removing unnecessary folder: {os.path.join(dest_dir, 'animals.zip')}")
        shutil.rmtree(os.path.join(dest_dir, "animals.zip"))
        print("Folder 'animals.zip' removed successfully.")

# Main execution
if __name__ == "__main__":
    # Download the dataset
    download_dataset(dataset_name, dataset_path)

    # Check if the file exists before trying to extract
    if os.path.exists(dataset_path):
        # Extract the dataset if it's a compressed file
        extract_dataset(dataset_path, raw_data_dir)
    else:
        print(f"Error: The file {dataset_path} does not exist or is not a valid file.")
