import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.preprocessing import StandardScaler

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def predict(self, X):
        return X.dot(self.w.T) + self.b

    def calculate_mse(self, y_actual, y_predicted):
        n = len(y_actual)
        squared_errors = (y_actual - y_predicted)**2
        mse = np.sum(squared_errors) / n
        return mse

    def train_linear_regression(self, X_train, y_train):
        m, p = X_train.shape  # Corrected: Use X_train.shape
        self.w = np.zeros((1, p))
        self.b = 0
        previous_mse = float('inf')
        learning_rate = self.learning_rate # Initialize learning_rate here, outside the conditional block

        for epoch in range(self.epochs):
            y_predicted = self.predict(X_train)
            errors = (y_train.reshape(-1, 1) - y_predicted)
            dw = (-2 / m) * np.dot(X_train.T, errors)
            db = (-2 / m) * np.sum(errors)

            self.w = self.w - learning_rate * dw.T # Use learning_rate (which is now properly scoped)
            self.b = self.b - learning_rate * db

            if epoch % 10 == 0:
                mse = self.calculate_mse(y_train, y_predicted)
                print(f"Epoch {epoch}, MSE: {mse}, w: {self.w}, b: {self.b}, LR: {learning_rate}")

                # Adaptive Learning Rate Logic:
                if mse >= previous_mse:
                    learning_rate *= 0.5 # Reduce learning rate
                    print(f"Learning rate reduced to: {learning_rate}")

                previous_mse = mse
                if learning_rate < 1e-9:
                    print("Learning rate too small, stopping training.")
                    break
        self.learning_rate = learning_rate

    def fit(self, X_train, y_train):  # Public fit method
        self.train_linear_regression(X_train, y_train)


if __name__ == '__main__':
    property_df = pd.read_csv("../linear-regression/dataset/train.csv")
    df_filtered = property_df.drop(columns=["size_units", "lot_size_units"])
    df_filtered.dropna(inplace=True)
    df = df_filtered.apply(pd.to_numeric, errors='coerce').dropna()

    X = df.iloc[:, :-1].values
    y = df["price"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    selector = SelectKBest(score_func=mutual_info_regression, k=3)
    X_new = selector.fit_transform(X_scaled, y)

    print("Selected feature indices:", selector.get_support(indices=True))

    model = LinearRegressionGD(learning_rate=0.01, epochs=1000) # Keep initial learning rate at 0.01, adaptive will adjust it
    model.fit(X_new, y)
    print(f"Model MSE: {model.calculate_mse(y, model.predict(X_new))}")