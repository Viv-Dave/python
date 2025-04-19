# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
# file_path = "../kmeans/dataset/stars.csv"
# input_data = pd.read_csv(file_path)
# input_filtered = input_data.drop(columns=["Star type", "Star color", "Spectral Class"])
# expected_output = input_data["Star type"]
# X_train, X_test = train_test_split(input_filtered, test_size=0.3, random_state=42)
# kmeans = KMeans(n_clusters=6, random_state=0, n_init=10)
# kmeans.fit(X_train)
# test_cluster_labels = kmeans.predict(X_test)
# test_cluster_labels = kmeans.predict(X_test)
# print("\nPredicted cluster labels for the first 10 test samples:")
# print(test_cluster_labels[:10])
# print("\nActual star types for the first 10 test samples:")
# print(y_test.values[:10]) # .values gets numpy array for easy slicing