import os
import random
import shutil

splitsize = 0.85
categories = []

Source_folder = "C:/Users/admin/Downloads/Bird_Dataset"
folders = os.listdir(Source_folder)
print(folders)

for subfolder in folders:
    if os.path.isdir(os.path.join(Source_folder, subfolder)):
        categories.append(subfolder)

categories.sort()
print(categories)


#create traget folder
target_folder = "C:/Users/admin/Downloads/Bird_Dataset/dataset_for_model"
if not os.path.exists(target_folder):
    os.mkdir(target_folder)


#creat a function to split the data for train and validation
def split_data(source, train, validation, split_size):
    files = []

    image_extensions = {'.png', '.jpg'}

    for root, _, filenames in os.walk(source):
        for filename in filenames:
            file = os.path.join(root, filename)
            if file.lower().endswith(tuple(image_extensions)):
                if os.path.getsize(file) > 0:
                    files.append(file)
                else:
                    print(f"{filename} is zero length, so ignoring.")
            else:
                print(f"{filename} is not an image file, so ignoring.")
    print(f"Total valid files: {len(files)}")

    trainingLength = int(len(files) * split_size)
    shuffleSet = random.sample(files, len(files))
    trainingSet = shuffleSet[:trainingLength]
    validSet = shuffleSet[trainingLength:]
    
    
# copy the train images
    for file in trainingSet:
        destination = os.path.join(train, os.path.relpath(file, source))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            shutil.copyfile(file, destination)
        except PermissionError as e:
            print(f"Permission denied for file: {file}. Error: {e}")

# copy the validation images
    for file in validSet:
        destination = os.path.join(validation, os.path.relpath(file, source))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        try:
            shutil.copyfile(file, destination)
        except PermissionError as e:
            print(f"Permission denied for file: {file}. Error: {e}")

trainPath = os.path.join(target_folder, "train")
validationPath = os.path.join(target_folder, "validation")


# create the target folder
if not os.path.exists(trainPath):
    os.mkdir(trainPath)

if not os.path.exists(validationPath):
    os.mkdir(validationPath)


# run the function for each of the folders
for category in categories:
    trainDesPath = os.path.join(trainPath, category)
    validationDesPath = os.path.join(validationPath, category)

    if not os.path.exists(trainDesPath):
        os.mkdir(trainDesPath)

    if not os.path.exists(validationDesPath):
        os.mkdir(validationDesPath)

    sourcePath = os.path.join(Source_folder, category)
    print(f"Copying from: {sourcePath} to: {trainDesPath} and {validationDesPath}")

    split_data(sourcePath, trainDesPath, validationDesPath, splitsize)
