# Commented out IPython magic to ensure Python compatibility.
# %pip install -U pip wheel setuptools
# %pip install concrete-ml

"""Importing Libraries"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

import sys
import requests
import json
from urllib.parse import urlparse
from PIL import Image

def download_image(url, folder):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    filename = os.path.basename(urlparse(url).path)
    filename_without_ext = os.path.splitext(filename)[0]
    filename_jpg = f"{filename_without_ext}.jpg"

    img = Image.open(response.raw)
    
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(os.path.join(folder, filename_jpg), 'JPEG')
    print("saved!")

def main(data):
    print("Starting")
    for item in data:
        url, cancer = item
        if cancer:
            folder = "Skin_Data/Cancer"
        else:
            folder = "Skin_Data/Non_Cancer"
        os.makedirs(folder, exist_ok=True)
        download_image(url, folder)

#AI

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('L')  # Convert to grayscale
        img = img.resize((256, 256))  # Resize the image to a fixed size (adjust as needed)
        img = np.array(img)  # Convert PIL image to NumPy array
        images.append(img)
    return images

if __name__ == "__main__":
    # Load the JSON string from the command-line argument
    data = json.loads(sys.argv[1])
    main(data)
    base_path = 'Skin_Data'

    cancer_folder = os.path.join(base_path, 'Cancer')
    non_cancer_folder = os.path.join(base_path, 'Non_Cancer')
    
    # Load and preprocess images from the directories
    cancer_images = load_images_from_folder(cancer_folder)
    non_cancer_images = load_images_from_folder(non_cancer_folder)

    # Create labels for the images
    cancer_labels = np.ones(len(cancer_images))  # 1 for cancer
    non_cancer_labels = np.zeros(len(non_cancer_images))  # 0 for non-cancer

    # Flatten the images
    def flatten_images(images):
        return np.array([image.flatten() for image in images])

    cancer_images_flattened = flatten_images(cancer_images)
    non_cancer_images_flattened = flatten_images(non_cancer_images)

    # Combine the flattened images and labels
    X = np.concatenate((cancer_images_flattened, non_cancer_images_flattened), axis=0)
    y = np.concatenate((cancer_labels, non_cancer_labels), axis=0)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the logistic regression model
    logistic_regression_model = LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(X_train, y_train)

    # Predict the target labels on the test set
    y_pred = logistic_regression_model.predict(X_test)

    # Evaluate the model's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    # Set the path to save the model
    model_filepath = 'logistic_regression_model_weights.joblib'

    # Save the model using joblib
    joblib.dump(logistic_regression_model, model_filepath)
