import os
import logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def convert(path, size=(64, 64)):
    image_list = []
    for filename in os.listdir(path):
        if filename.lower().endswith("jpg"):
            image = Image.open(os.path.join(path, filename)).convert("L").resize(size)  
            image_array = np.array(image).flatten() 
            image_list.append(image_array)
    return np.array(image_list)

circle_path = r"D:\python\machine-learning\Support-Vector-Machines\datasets\circle"
square_path = r"D:\python\machine-learning\Support-Vector-Machines\datasets\square"

circle_array = convert(circle_path)
square_array = convert(square_path)

circle_df, square_df = pd.DataFrame(circle_array), pd.DataFrame(square_array)
circle_df["shape"], square_df["shape"] = 0, 1

df = pd.concat([circle_df, square_df], axis=0).reset_index(drop=True)

X, y = df.drop("shape", axis=1), df["shape"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

pca = PCA(n_components=500)
X_train_pca, X_test_pca = pca.fit_transform(X_train), pca.transform(X_test)
model = make_pipeline(StandardScaler(), SVC(kernel="linear", verbose=True))

logging.info("Starting model training...")
model.fit(X_train_pca, y_train)
logging.info("Model training completed.")

logging.info("Making predictions on the test set...")
predictions = model.predict(X_test_pca)

accuracy = accuracy_score(y_test, predictions)
logging.info(f"Model Accuracy: {accuracy*100:.2f}%")
print(f"Model Accuracy: {accuracy*100:.2f}%")

def process(image_path, size=(64, 64)):
    image = Image.open(image_path).convert("L").resize(size)  
    numpy_image = np.array(image)
    flattened = numpy_image.flatten().reshape(1, -1)  
    return flattened

test_image = process(r"D:\python\machine-learning\Support-Vector-Machines\datasets\dataset\test\circle\circle-2000.jpg")
test_image_2 = process(r"D:\python\machine-learning\Support-Vector-Machines\datasets\square\square-2.jpg")
test_image_pca = pca.transform(test_image)
test_image_2_pca =pca.transform(test_image_2)
prediction = model.predict(test_image_pca)[0]
prediction_2 = model.predict(test_image_2_pca)[0]
shape = "Circle" if prediction == 0 else "Square"
shape_2 = "Circle" if prediction_2 == 0 else "Square"
logging.info(f"Predicted Shape: {shape}")
logging.info(f"Predicted Shape for another test image: {shape_2}")