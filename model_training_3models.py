import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import ResNet50, DenseNet121, MobileNetV3Large
from tensorflow.keras.applications import MobileNetV3Small
# from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import AUC
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score


from model_evaluation import evaluate_model

# Verify Pillow installation
from PIL import Image
print("Pillow is installed and imported successfully!")

def initialise_elements():
    # Define image dimensions and batch size
    model_type = sys.argv[1] if len(sys.argv) > 1 else "resnet50"  # resnet50 // dense121 // mobilenetv3
    if model_type == "mobilenetv3":
        img_height, img_width = 224, 224
    else:
        img_height, img_width = 256, 256
    batch_size = 32
    epochs = 50 # 50
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

train_generator, validation_generator, test_generator, number_classes = initialise_dataset()

def train_model(base_model):
    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
    x = Dropout(0.5)(x)  # Add dropout for regularization
    predictions = Dense(number_classes, activation='softmax')(x)  # Output layer with correct number of classes

    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze more layers initially and then unfreeze for fine-tuning
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy', AUC(name='mAP')])

    # Add callbacks for early stopping and reducing learning rate on plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

    # Measure training time
    start_time = time.time()
    
    # Train the model with initial frozen layers
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Unfreeze the last few layers of the base model
    if model_type != "mobilenetv3": 
        for layer in base_model.layers[-10:]:
            layer.trainable = True

    # Recompile the model for these modifications to take effect
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy', AUC(name='mAP')])

    # Continue training the model with unfrozen layers
    history_fine_tune = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Combine histories
    for key in history.history.keys():
        history.history[key] += history_fine_tune.history[key]

    # Measure training time
    end_time = time.time()

    # Evaluate the model
    test_loss, test_accuracy, test_mAP = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test mAP: {test_mAP}")
    print(f"Test loss: {test_loss}")
    print(f"Training time: {end_time - start_time} seconds")

    # Save the trained model
    model_name = f'{model_type}_{number_classes}_classes_1.h5'
    model.save('models/' + model_name)
    return history, test_accuracy, test_mAP, end_time - start_time, model_name, model

def visualize_results(history, model_type, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{model_type} Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{model_type} Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.savefig(f'figs/{model_name}_performance.png')
    plt.show()

def plot_confusion_matrix(test_generator, model, model_name):
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
    plt.savefig(f'figs/{model_name}_confusion_matrix.png')
    plt.show()


if model_type == "resnet50":
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
elif model_type == "dense121":
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
elif model_type == "mobilenetv3":
    # print("Not implemented Yet")
    # raise ValueError("Not implemented yet")
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
else:
    raise ValueError("Invalid model type specified. Choose from 'resnet50', 'dense121', or 'mobilenetv3'.")

history, test_accuracy, test_mAP, training_time, model_name, model = train_model(base_model)
visualize_results(history, model_type, model_name)
plot_confusion_matrix(test_generator, model, model_name)
evaluate_model(test_generator, model_name)