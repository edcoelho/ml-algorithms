import numpy as np

class OLSLinearRegressor ():
    """
        A class used to represent an regressor based on Ordinary Least Square method.
    """

    def __init__ (self):
        self.is_fitted = False

    def fit (self, X: np.ndarray, y: np.ndarray):
        X_transpose_view = X.transpose()
        self.w = np.linalg.inv(X_transpose_view @ X) @ X_transpose_view @ y

    def predict (self, X: np.ndarray):
        return X @ self.w