import os
import sys
from tensorflow.keras.applications import ResNet50, DenseNet121
# from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large, MobileNetV3Small
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Verify Pillow installation
from PIL import Image
print("Pillow is installed and imported successfully!")

def initialise_elements():
    # Define image dimensions and batch size
    img_height, img_width = 256, 256
    batch_size = 32
    epochs = 50
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

    # Unfreeze the last few layers of the base model
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    # Recompile the model for these modifications to take effect
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Add callbacks for early stopping and reducing learning rate on plateau
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    # Train the model
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping, reduce_lr]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy: {test_accuracy}")
    print(f"Test loss: {test_loss}")

    # Save the trained model
    model.save(f'{model_type}_{number_classes}_model.h5')

def train_module_resnet50():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    train_model(base_model)

def train_module_dense121():
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    train_model(base_model)

def train_module_mobilenetv3():
    base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    train_model(base_model)

if model_type == "resnet50":
    train_module_resnet50()
elif model_type == "dense121":
    train_module_dense121()
elif model_type == "mobilenetv3":
    print("Not implemented yet")
    # train_module_mobilenetv3()
