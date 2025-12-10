import numpy as np

def train_test_split(X: np.ndarray, y: np.ndarray, train_size: float, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the data matrix (features) and the target values array into random train and test subsets.
    
    :param X: The data matrix (features).
    :param y: The **target values array** (labels).
    :param train_size: The **proportion** of the dataset to include in the train split. Should be a float between 0.0 and 1.0.
    :param shuffle: Whether or not to shuffle the data before splitting.
    :returns: A tuple containing the split data: (X_train, X_test, y_train, y_test).
    """

    indices = np.random.permutation(X.shape[0]) if shuffle else np.arange(X.shape[0])
    n_train = int(np.floor(indices.shape[0] * train_size))
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]