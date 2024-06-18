import os
import zipfile
import random
from PIL import Image
import numpy as np

# Define parameters
zip_path = 'C:/Users/Victus/Desktop/DataPreparation/ShuffledData.zip'
extract_to = 'dataset'
base_dir = 'data_split'
categories = ['Parrot', 'Eagle', 'Hornbill', 'Duck', 'Owl']

# Function to extract dataset
def extract_dataset(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# Function to create folders for dataset splits
def create_folders(base_dir, categories):
    for split in ['training', 'validation', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for category in categories:
            os.makedirs(os.path.join(split_dir, category), exist_ok=True)

# Function to resize and standardize images
def process_image(image_path, size=(224, 224)):
    image = Image.open(image_path)
    image = image.resize(size)
    image = np.array(image) / 255.0  # Standardizing the data
    return Image.fromarray((image * 255).astype(np.uint8))  # Convert back to image format

# Function to distribute images into respective folders
def distribute_images(dataset_path, base_dir, categories, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    for category in categories:
        image_files = [f for f in os.listdir(dataset_path) if f.startswith(category)]
        random.shuffle(image_files)
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'training': image_files[:n_train],
            'validation': image_files[n_train:n_train + n_val],
            'test': image_files[n_train + n_val:]
        }

        for split, files in splits.items():
            for file_name in files:
                src_path = os.path.join(dataset_path, file_name)
                dest_path = os.path.join(base_dir, split, category, file_name)
                processed_image = process_image(src_path)
                processed_image.save(dest_path)

# Main function to execute the process
def main(zip_path, extract_to, base_dir, categories):
    # Step 1: Extract dataset
    extract_dataset(zip_path, extract_to)

    # Step 2: Create folders
    create_folders(base_dir, categories)

    # Step 3: Distribute images and process them
    distribute_images(extract_to, base_dir, categories)

# Execute main function
main(zip_path, extract_to, base_dir, categories)
