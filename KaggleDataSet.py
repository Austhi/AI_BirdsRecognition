import os
from kaggle.api.kaggle_api_extended import KaggleApi
import zipfile

# Set up the custom path for Kaggle configuration
os.environ['KAGGLE_CONFIG_DIR'] = r'C:\Users\Victus\.kaggle'

# Initialize and authenticate the Kaggle API
api = KaggleApi()
api.authenticate()

# Search for datasets related to bird images
datasets = api.dataset_list(search='bird images')
for dataset in datasets:
    print(f"{dataset.title}: {dataset.ref}")

# Example dataset reference (you can choose any dataset from the search results)
dataset_ref = 'gpiosenka/100-bird-species'
download_folder = 'kaggle_datasets'

# Create a folder to store the dataset
if not os.path.exists(download_folder):
    os.makedirs(download_folder)

# Download the dataset
api.dataset_download_files(dataset_ref, path=download_folder, unzip=True)

# Unzip dataset files if necessary
zip_file_path = os.path.join(download_folder, f"{dataset_ref.split('/')[-1]}.zip")
if os.path.exists(zip_file_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_folder)
