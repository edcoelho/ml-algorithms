import numpy as np
from ml_algos.metrics import mean_squared_error

class OLSLinearRegression():
    """
        A class that represents an linear regressor based on Ordinary Least Square method.
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_transpose_view = X.transpose()
        self.w = np.linalg.inv(X_transpose_view @ X) @ X_transpose_view @ y
        self.is_fitted = True

    def predict(self, X: np.ndarray):
        if self.is_fitted:
            return X @ self.w
        else:
            raise("The OLSLinearRegression model is not fitted.")
    
class GDLinearRegression():
    """
        A class that represents an linear regressor based on Gradient Descent algorithm.
    """

    def __init__(self, epochs = 10000, learning_rate = 1e-2):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros((X.shape[1], 1))
        self.train_loss_history = []

        for _ in range(self.epochs):
            y_hat = X @ self.w
            train_loss, error = mean_squared_error(y, y_hat, True)
            self.train_loss_history.append(train_loss)
            self.w = self.w + self.learning_rate*(X.transpose() @ error)/X.shape[0]

        self.is_fitted = True


    def predict(self, X: np.ndarray):
        if self.is_fitted:
            return X @ self.w
        else:
            raise("The GDLinearRegression model is not fitted.")

class SGDLinearRegression():
    """
        A class that represents an linear regressor based on Stochastic Gradient Descent algorithm.
    """

    def __init__(self, epochs = 1000, learning_rate = 1e-2):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.w = np.zeros((X.shape[1], 1))
        self.train_loss_history = []
        y_hat = X @ self.w

        for _ in range(self.epochs):
            indices = np.random.permutation(X.shape[0])

            for i in indices:
                y_hat[i] = X[i].reshape(1, -1) @ self.w
                train_loss, error = mean_squared_error(y, y_hat, True)
                self.train_loss_history.append(train_loss)
                self.w = self.w + self.learning_rate*error[i]*X[i].reshape(-1, 1)

        self.is_fitted = True

    def predict(self, X: np.ndarray):
        if self.is_fitted:
            return X @ self.w
        else:
            raise("The SGDLinearRegression model is not fitted.")