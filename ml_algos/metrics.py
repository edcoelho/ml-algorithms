import numpy as np

def mean_squared_error(y: np.ndarray, y_hat: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Function to compute mean squared error loss.
    
    :param y: Target values
    :param y_hat: Predicted values
    """

    raw_error = y - y_hat
    loss = np.sum(raw_error**2) / (2*y.shape[0])

    return loss, raw_error