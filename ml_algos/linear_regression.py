import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from ml_algos.metrics import mean_squared_error

class GradientModel(ABC):
    """
    A class that represents an generic model based on gradient.
    """

    def __init__(self, l2_reg_term: float, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]) -> None:
        super().__init__()
        self._l2_reg_term = l2_reg_term
        self._loss_function = loss_function
        self._is_fitted = False

    def get_w(self) -> np.ndarray:
        if self._is_fitted:
            return self._w
        else:
            raise(f"The {self.__class__.__name__} model is not fitted.")
        
    def get_l2_reg_term(self) -> float:
        return self._l2_reg_term
    
    def set_l2_reg_term(self, l2_reg_term) -> None:
        self._l2_reg_term = l2_reg_term

    def set_loss_function(self, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]]) -> None:
        self._loss_function = loss_function

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True) -> None:
        pass

    def predict(self, X: np.ndarray, include_bias: bool = True) -> np.ndarray:
        if self._is_fitted:
            X_prime = X
            if include_bias:
                X_prime = np.column_stack((np.ones((X.shape[0], 1)), X))
            return X_prime @ self._w
        else:
            raise Exception(f"The {self.__class__.__name__} model is not fitted.")

class OLSLinearRegression(GradientModel):
    """
    A class that represents an linear regressor based on Ordinary Least Square method.
    """

    def __init__(self, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error) -> None:
        super().__init__(l2_reg_term, loss_function)
        self._l2_reg_term = l2_reg_term

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True) -> None:
        X_prime = X
        if include_bias:
            X_prime = np.column_stack((np.ones((X.shape[0], 1)), X))

        X_prime_transpose = X_prime.transpose()
        l2_reg_term_matrix = self._l2_reg_term * np.identity(X_prime.shape[1])
        self._w = np.linalg.inv(X_prime_transpose @ X_prime + l2_reg_term_matrix) @ X_prime_transpose @ y
        self._is_fitted = True
    
class GDLinearRegression(GradientModel):
    """
    A class that represents an linear regressor based on Gradient Descent algorithm.
    """

    def __init__(self, epochs: int = 10000, learning_rate: float = 1e-2, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error) -> None:
        super().__init__(l2_reg_term, loss_function)
        self._epochs = epochs
        self._learning_rate = learning_rate

    def get_epochs(self) -> int:
        return self._epochs

    def set_epochs(self, epochs: int) -> None:
        self._epochs = epochs

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        self._learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True) -> None:
        X_prime = X
        if include_bias:
            X_prime = np.column_stack((np.ones((X.shape[0], 1)), X))

        self._w = np.zeros((X_prime.shape[1], 1))
        self.train_loss_history = []

        for _ in range(self._epochs):
            y_hat = X_prime @ self._w
            train_loss, raw_error = self._loss_function(y, y_hat)
            self.train_loss_history.append(train_loss)
            self._w = self._w + self._learning_rate * (((X_prime.transpose() @ raw_error) / X_prime.shape[0]) - self._l2_reg_term * self._w)

        self._is_fitted = True

class SGDLinearRegression(GradientModel):
    """
    A class that represents an linear regressor based on Stochastic Gradient Descent algorithm.
    """

    def __init__(self, epochs: int = 100, learning_rate: float = 1e-2, l2_reg_term: float = 0.0, loss_function: Callable[[np.ndarray, np.ndarray], tuple[float, np.ndarray]] = mean_squared_error) -> None:
        super().__init__(l2_reg_term, loss_function)
        self._epochs = epochs
        self._learning_rate = learning_rate

    def get_epochs(self) -> int:
        return self._epochs

    def set_epochs(self, epochs: int) -> None:
        self._epochs = epochs

    def get_learning_rate(self) -> float:
        return self._learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        self._learning_rate = learning_rate

    def fit(self, X: np.ndarray, y: np.ndarray, include_bias: bool = True) -> None:
        X_prime = X
        if include_bias:
            X_prime = np.column_stack((np.ones((X.shape[0], 1)), X))

        self._w = np.zeros((X_prime.shape[1], 1))
        self.train_loss_history = []
        y_hat = X_prime @ self._w

        for _ in range(self._epochs):
            indices = np.random.permutation(X_prime.shape[0])

            for i in indices:
                y_hat[i] = X_prime[i].reshape(1, -1) @ self._w
                train_loss, raw_error = self._loss_function(y, y_hat)
                self.train_loss_history.append(train_loss)
                self._w = self._w + self._learning_rate * (raw_error[i] * X_prime[i].reshape(-1, 1) - self._l2_reg_term * self._w)

        self._is_fitted = True