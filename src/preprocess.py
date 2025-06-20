# src/preprocess.py

import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    assert img is not None, f"Failed to load image: {image_path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img.flatten()

def load_dataset(data_dir):
    image_paths = []
    labels = []

    for label_dir in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label_dir)
        if os.path.isdir(label_path):
            for img_name in os.listdir(label_path):
                image_paths.append(os.path.join(label_path, img_name))
                labels.append(label_dir)

    df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    return df

def preprocess_data(df, test_size=0.2, val_size=0.25):
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])
    num_classes = len(label_encoder.classes_)

    X = np.array([load_and_preprocess_image(path) for path in df['image_path']])
    y = df['label_encoded'].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42)

    one_hot = OneHotEncoder(sparse_output=False)
    y_train_one_hot = one_hot.fit_transform(y_train.reshape(-1, 1))
    y_val_one_hot = one_hot.transform(y_val.reshape(-1, 1))
    y_test_one_hot = one_hot.transform(y_test.reshape(-1, 1))

    return X_train, X_val, X_test, y_train_one_hot, y_val_one_hot, y_test_one_hot, num_classes, label_encoder