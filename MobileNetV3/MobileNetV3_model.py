import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV3Small, MobileNetV3Large
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from sklearn.metrics import average_precision_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# custom callback to calculate mean Average Precision (mAP)
class MAPCallback(Callback):
    def __init__(self, validation_data, num_classes):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes

    def on_epoch_end(self, epoch, logs=None):
        val_gen = self.validation_data
        val_gen.reset()
        y_true = []
        y_pred = []
        for _ in range(len(val_gen)):
            x_val, y_val = val_gen.__next__()
            y_true.extend(y_val)
            y_pred.extend(self.model.predict(x_val))
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        average_precisions = []
        for i in range(self.num_classes):
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            if np.any(y_true_i):
                average_precisions.append(average_precision_score(y_true_i, y_pred_i))
            else:
                average_precisions.append(0.0)

        mAP = np.mean(average_precisions)
        print(f"mAP: {mAP}")
        logs['mAP'] = mAP


# path for the train and validation folders
trainPath = "dataset/train"
validPath = "dataset/val"


# image augmentation
trainGenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.8, 1.2),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    channel_shift_range=0.1,
    rescale=1./255
).flow_from_directory(
    trainPath, target_size=(320, 320), batch_size=32, class_mode='categorical'
)

validGenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.8, 1.2),
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    channel_shift_range=0.1,
    rescale=1./255
).flow_from_directory(
    validPath, target_size=(320, 320), batch_size=32, class_mode='categorical'
)

    
    

# build the model
baseModel = MobileNetV3Small(weights="imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)

num_classes = len(trainGenerator.class_indices)
predictionLayer = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=baseModel.input, outputs=predictionLayer)

print(model.summary())


# freeze the layer of the model already trained
for layer in baseModel.layers:
    layer.trainable = False



# compile
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])



lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
mAP_callback = MAPCallback(validation_data=validGenerator, num_classes=num_classes)

# train
history = model.fit(trainGenerator, validation_data=validGenerator, epochs=50, callbacks=[lr_scheduler, mAP_callback])


# saving the model
modelSavedPath = "C:/Users/admin/Downloads/Bird_Dataset/dataset_for_model/model.keras"
model.save(modelSavedPath)


# visualisation
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()


y_true = []
y_pred = []

validGenerator.reset()
for _ in range(len(validGenerator)):
    x_val, y_val = validGenerator.__next__()
    y_true.extend(np.argmax(y_val, axis=1))
    y_pred.extend(np.argmax(model.predict(x_val), axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)



cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(trainGenerator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues)
plt.show()



print(classification_report(y_true, y_pred, target_names=list(trainGenerator.class_indices.keys())))
