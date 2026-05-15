import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

PATH = "../knn/datasets/dataset.csv"
df = pd.read_csv(PATH)

mlb = MultiLabelBinarizer()
label = LabelEncoder()
# print(df.head())
X = df.iloc[:, df.columns != 'Disease']
y = df["Disease"]

symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
df[symptom_cols] = df[symptom_cols].fillna('')
df['Symptoms'] = df[symptom_cols].apply(lambda row: ', '.join(row), axis=1)
df = df[['Disease', 'Symptoms']]
df['Symptoms'].nunique()

print(X.head())
print(y.head())
X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.size, y_train.size, X_test.size, y_test.size)
model = RandomForestClassifier()

model.fit(X_train, y_train)
train_proba = model.predict_proba(X_train)
test_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")