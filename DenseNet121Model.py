import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
test_dir = 'data_split/test'
num_classes = 5  # Number of classes (Parrot, Eagle, Hornbill, Duck, Owl)

# Create ImageDataGenerator instance for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical')

# Load the saved model
model_path = os.path.join('DenseNet121_Model', 'densenet121_model.h5')
model = load_model(model_path)

# Evaluate the model
loss, accuracy = model.evaluate(test_gen)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

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
plt.show()

# Confusion matrix
test_gen.reset()
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=-1)
cm = confusion_matrix(y_true, y_pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# Calculate mAP (mean average precision)
y_true_binary = np.zeros((len(y_true), num_classes))
for i, label in enumerate(y_true):
    y_true_binary[i, label] = 1
y_pred_proba = model.predict(test_gen)
mAP = average_precision_score(y_true_binary, y_pred_proba, average='macro')
print(f'mAP (mean Average Precision): {mAP}')
