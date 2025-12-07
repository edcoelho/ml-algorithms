import numpy as np

def mean_squared_error(y: np.ndarray, y_hat: np.ndarray, return_error = False):
    """
    Function to compute mean squared error loss.
    
    :param y: Target values
    :param y_hat: Predicted values
    :param return_error: If True, returns the raw error values and the loss. Otherwise, returns only the loss.
    """

    error = y - y_hat
    loss = np.sum(error**2) / (2*y.shape[0])

    if return_error:
        return loss, error
    else:
        return loss