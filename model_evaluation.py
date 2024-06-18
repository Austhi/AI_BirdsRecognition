from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def initialise_elements():
    # Define image dimensions and batch size
    img_height, img_width = 256, 256
    batch_size = 32
    epochs = 1 # 50
    model_type = sys.argv[1] if len(sys.argv) > 1 else "resnet50"  # resnet50 // dense121 // mobilenetv3
    dataset_folder = 'dataset/'

    return img_height, img_width, batch_size, epochs, model_type, dataset_folder

img_height, img_width, batch_size, epochs, model_type, dataset_folder = initialise_elements()

def initialise_dataset():
    # Define directory paths
    train_data_dir = os.path.join(dataset_folder, 'train')
    validation_data_dir = os.path.join(dataset_folder, 'val')
    test_data_dir = os.path.join(dataset_folder, 'test')

    number_classes = len(os.listdir(train_data_dir))
    num_validation_classes = len(os.listdir(validation_data_dir))

    print(f'Number of training classes: {number_classes}')
    print(f'Number of validation classes: {num_validation_classes}')

    # Create data generators for training, validation, and testing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Create the data generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Usually for test data, we don't shuffle
    )
    return train_generator, validation_generator, test_generator, number_classes

def plot_confusion_matrix(test_generator, model_name):
    if model_name == "":
        return
    print(model_name)
    model = load_model(model_name)
    
    path = model_name.split('/')
    name = path[len(path) - 1]

    # Predict
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(test_generator.class_indices.keys()), yticklabels=list(test_generator.class_indices.keys()))
    plt.title('Confusion Matrix')
    plt.ylabel('Actual class')
    plt.xlabel('Predicted class')
    plt.savefig(f'figs/{name}_confusion_matrix.png')
    plt.show()

def evaluate_model(test_generator, model_name):
    # Load the model
    model = load_model(model_name)

    path = model_name.split('/')
    name = path[len(path) - 1]

    # Predict
    Y_pred = model.predict(test_generator)
    y_pred = np.argmax(Y_pred, axis=1)
    y_true = test_generator.classes

    print(f"y_pred shape: {y_pred.shape}")
    print(f"y_true shape: {y_true.shape}")

    print('Classification Report')
    print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

if __name__ == '__main__':
    if len(sys.argv) > 0:
        train_generator, validation_generator, test_generator, number_classes = initialise_dataset()
        plot_confusion_matrix(test_generator, sys.argv[1])
        evaluate_model(test_generator, sys.argv[1])