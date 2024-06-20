import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define directories
test_dir = 'dataset/test'
graph_dir = 'figs'

# Create ImageDataGenerator instance for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Load the saved model
model_path = os.path.join('Models', 'densenet121_model.h5')
model = load_model(model_path)

# Evaluate the model
loss, accuracy = model.evaluate(test_gen)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Confusion matrix
test_gen.reset()
y_true = test_gen.classes
y_pred = np.argmax(model.predict(test_gen), axis=-1)
print("Class indices from generator:", test_gen.class_indices)
class_names = list(test_gen.class_indices.keys())
print("Class names:", class_names)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i, j]), horizontalalignment='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(graph_dir, 'dense121_5_model.h5_confusion_matrix.png'))
plt.show()

# Classification report
print('Classification Report')
print(classification_report(y_true, y_pred, target_names=class_names))
