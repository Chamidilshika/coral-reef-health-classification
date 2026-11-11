import cv2
import os
import numpy as np

IMG_SIZE = 224

def load_data(folder):
    images = []
    labels = []

    for category in ["healthy", "bleached"]:
        path = os.path.join(folder, category)

        label = 0 if category == "healthy" else 1

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalization

            images.append(img)
            images.append(label)

    return np.array(images),np.array(labels)