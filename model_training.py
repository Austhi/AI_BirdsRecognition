import os
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Verify Pillow installation
from PIL import Image
print("Pillow is installed and imported successfully!")

# Define directory paths
train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/val'
test_data_dir = 'dataset/test'


num_train_classes = len(os.listdir(train_data_dir))
num_validation_classes = len(os.listdir(validation_data_dir))

print(f'Number of training classes: {num_train_classes}')
print(f'Number of validation classes: {num_validation_classes}')

# Define image dimensions and batch size
img_height, img_width = 256, 256
batch_size = 32

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

# Load the ResNet50 model with pre-trained ImageNet weights, excluding the top layer
base_model = MobileNetV3Large(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Add custom layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)  # Add a fully connected layer
x = Dropout(0.5)(x)  # Add dropout for regularization
predictions = Dense(3, activation='softmax')(x)  # Output layer with correct number of classes


# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze the last few layers of the base model
for layer in base_model.layers[-10:]:
    layer.trainable = True


# Recompile the model for these modifications to take effect
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 50

# Add callbacks for early stopping and reducing learning rate on plateau
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

# Train again (this time fine-tuning the top 10 layers of ResNet50)
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# Save the trained model
model.save('resnet50_3_classes.h5')