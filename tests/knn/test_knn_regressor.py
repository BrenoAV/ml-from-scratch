from sklearn.datasets import make_regression
import numpy as np

import pytest

from src.knn import KNNRegressor


rng = np.random.default_rng(123)


@pytest.fixture(scope="session")
def dataset_2d():
    X, y = make_regression(n_samples=10, n_features=2, random_state=123)
    return X, y


@pytest.fixture(scope="session")
def dataset_10d():
    X, y = make_regression(n_samples=20, n_features=10, random_state=123)
    return X, y


# Tests to verify if change a little the position will be return the correct
def test_knn_predict_dataset_2d_and_k_1(dataset_2d):
    X, y = dataset_2d
    model = KNNRegressor(X, y)

    x0 = X[0] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x0, 1) == y[0]
    x1 = X[1] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x1, 1) == y[1]
    x2 = X[2] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x2, 1) == y[2]


def test_knn_predict_dataset_10d_and_k_1(dataset_10d):
    X, y = dataset_10d
    model = KNNRegressor(X, y)
    x0 = X[0] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x0, 1) == y[0]
    x1 = X[1] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x1, 1) == y[1]
    x2 = X[2] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x2, 1) == y[2]
    x3 = X[4] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x3, 1) == y[4]
    x4 = X[7] + rng.normal(loc=0, scale=0.01)
    assert model.predict(x4, 1) == y[7]


def test_knn_predict_dataset_2d_and_k_3():
    X = np.array([[0, 0], [-1, 1], [1, 1], [2, 0], [-1, 2]])
    y = np.array([3.1, 2.5, 4.3, 1.1, 0.5])
    x1 = np.array([0, 2])
    x2 = np.array([3, -3])
    x3 = np.array([0, 1])
    model = KNNRegressor(X, y)
    assert model.predict(x1, 3) == np.mean([0.5, 2.5, 4.3])
    assert model.predict(x2, 3) == np.mean([3.1, 4.3, 1.1])
    assert model.predict(x3, 3) == np.mean([2.5, 3.1, 4.3])
