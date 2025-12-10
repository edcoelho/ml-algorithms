from typing import Callable
from itertools import combinations, combinations_with_replacement
from abc import ABC
import numpy as np
from ml_algos.linear_regression import OLSLinearRegression, GDLinearRegression, SGDLinearRegression
from ml_algos.metrics import mean_squared_error

def add_polynomial_features(X: np.ndarray, degree: int, include_bias: bool = True, interaction_only: bool = False) -> np.ndarray:
    """
    Generates a new feature matrix containing all polynomial terms of the original features up to the specified degree.
    
    This includes powers of the original features (e.g., X1^2) and interaction terms between them (e.g., X1 * X2).
    
    :param X: Matrix to be transformed.
    :param degree: Polynomial degree
    :param include_bias: If True, includes a bias column in the matrix.
    :return: Transformed matrix.
    """

    X_transformed = X
    new_features = []

    combiner = combinations if interaction_only else combinations_with_replacement

    for d in range(2, degree + 1):
        for i in combiner(range(X.shape[1]), d):
            new_features.append(np.prod(X[:, i], axis=1))

    if include_bias:
        X_transformed = np.column_stack((np.ones((X_transformed.shape[0], 1)), X_transformed))

    if new_features:
        new_features = np.column_stack(new_features)
        X_transformed = np.column_stack((X_transformed, new_features))

    return X_transformed

class PolynomialRegressorL2(ABC):
    """
    A class that represents an generic polynomial regressor with L2 regularization.
    """

    def __init__(self, degree: int):
        super().__init__()
        self._degree = degree

    def get_degree(self):
        return self._degree
    
    def set_degree(self, degree: int):
        self._degree = degree
    
class OLSPolynomialRegressionL2(PolynomialRegressorL2, OLSLinearRegression):
    """
    A class that represents an polynomial regressor based on Ordinary Least Square method with L2 regularization.
    """

    def __init__(self, degree: int, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error):
        super().__init__(degree = degree, l2_reg_term = l2_reg_term, loss_function = loss_function)

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True):
        X_prime = add_polynomial_features(X, self._degree, include_bias)

        return super().fit(X_prime, y, False)

    def predict(self, X, include_bias = True):
        X_prime = add_polynomial_features(X, self._degree, False)

        return super().predict(X_prime, include_bias)

class GDPolynomialRegressionL2(PolynomialRegressorL2, GDLinearRegression):
    """
    A class that represents an polynomial regressor based on Gradient Descent method with L2 regularization.
    """

    def __init__(self, degree: int, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error):
        super().__init__(degree = degree, l2_reg_term = l2_reg_term, loss_function = loss_function)

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True):
        X_prime = add_polynomial_features(X, self._degree, include_bias)

        return super().fit(X_prime, y, False)

    def predict(self, X, include_bias = True):
        X_prime = add_polynomial_features(X, self._degree, False)

        return super().predict(X_prime, include_bias)

class SGDPolynomialRegressionL2(PolynomialRegressorL2, SGDLinearRegression):
    """
    A class that represents an polynomial regressor based on Stochastic Gradient Descent method with L2 regularization.
    """

    def __init__(self, degree: int, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error):
        super().__init__(degree = degree, l2_reg_term = l2_reg_term, loss_function = loss_function)

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True):
        X_prime = add_polynomial_features(X, self._degree, include_bias)

        return super().fit(X_prime, y, False)

    def predict(self, X, include_bias = True):
        X_prime = add_polynomial_features(X, self._degree, False)

        return super().predict(X_prime, include_bias)