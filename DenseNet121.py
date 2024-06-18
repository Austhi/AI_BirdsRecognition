import os
import random
import zipfile
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create directories for saving graphs and the model
graph_dir = 'DenseNet121_Graph'
model_dir = 'DenseNet121_Model'
os.makedirs(graph_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Define directories
train_dir = 'data_split/training'
val_dir = 'data_split/validation'
test_dir = 'data_split/test'
num_classes = 5  # Number of classes (Parrot, Eagle, Hornbill, Duck, Owl)

# Create ImageDataGenerator instances for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')
test_gen = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load DenseNet121 base model
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine base model and custom layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Record the training time
start_time = time.time()

# Train the model
history = model.fit(train_gen, validation_data=val_gen, epochs=2)

# Calculate training time
training_time = time.time() - start_time
print(f"Training Time: {training_time} seconds")

# Save the trained model
model.save(os.path.join(model_dir, 'densenet121_model.h5'))

# Evaluate the model
loss, accuracy = model.evaluate(test_gen)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Plot training history (accuracy and loss)
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim([0, max(history.history['loss'])])
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.savefig(os.path.join(graph_dir, 'accuracy_loss.png'))
plt.show()

# Visualize some results
class_names = sorted(test_gen.class_indices.keys())
test_images, test_labels = next(test_gen)
predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.imshow(test_images[i])
    true_label = class_names[np.argmax(test_labels[i])]
    predicted_label = class_names[np.argmax(predictions[i])]
    plt.title(f'True: {true_label}, Predicted: {predicted_label}')
    plt.axis('off')
plt.savefig(os.path.join(graph_dir, 'sample_predictions.png'))
plt.show()

# Confusion matrix
test_gen.reset()
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=-1)
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
plt.show()

# Calculate mAP (mean average precision)
y_true_binary = np.zeros((len(y_true), num_classes))
for i, label in enumerate(y_true):
    y_true_binary[i, label] = 1
y_pred_proba = model.predict(test_gen)
mAP = average_precision_score(y_true_binary, y_pred_proba, average='macro')
print(f'mAP (mean Average Precision): {mAP}')
