import os
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2


train_path = "C:/Users/admin/Downloads/Bird_Dataset/dataset_for_model/train"
categories = sorted(os.listdir(train_path))
print(categories)


modelSavedPath = "C:/Users/admin/Downloads/Bird_Dataset/dataset_for_model/model.keras"
model = tf.keras.models.load_model(modelSavedPath)


def classify_image(imageFile):
    img = Image.open(imageFile)
    img = img.resize((320, 320), Image.LANCZOS)
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  
    
    print(f"Input image shape: {x.shape}")
    pred = model.predict(x)
    print(f"Raw model output: {pred}")
    
    class_idx = np.argmax(pred, axis=1)[0]
    class_name = categories[class_idx]
    print(f"Predicted class: {class_name}")
    
   
    categoryValue = np.argmax(pred, axis=1)
    categoryValue = categoryValue[0]
    
    print(categoryValue)
    
    result = categories[categoryValue]
    
    return result


img_path = "D:/Download/Images/owlTest.png"
resultText = classify_image(img_path)
print(resultText)


img = cv2.imread(img_path)
img = cv2.putText(img, resultText, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()