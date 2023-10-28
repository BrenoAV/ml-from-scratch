import numpy as np
import pytest
from src.knn import KNNClassifier
from sklearn.datasets import make_blobs

rng = np.random.default_rng(123)


@pytest.fixture(scope="session")
def dataset_2d():
    X, y = make_blobs(
        n_samples=9, cluster_std=0.4, centers=3, n_features=2, random_state=123
    )
    return X, y


@pytest.fixture(scope="session")
def dataset_10d():
    X, y = make_blobs(
        n_samples=15, cluster_std=0.4, centers=5, n_features=10, random_state=123
    )
    return X, y


def test_knn_predict_dataset_2d_and_k_1(dataset_2d):
    X, y = dataset_2d
    model = KNNClassifier(X, y)

    x0 = X[0] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x0, 1) == y[0]
    x1 = X[1] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x1, 1) == y[1]
    x2 = X[2] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x2, 1) == y[2]


def test_knn_predict_dataset_10d_and_k_1(dataset_10d):
    X, y = dataset_10d
    model = KNNClassifier(X, y)
    x0 = X[0] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x0, 1) == y[0]
    x1 = X[1] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x1, 1) == y[1]
    x2 = X[2] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x2, 1) == y[2]
    x3 = X[4] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x3, 1) == y[4]
    x4 = X[7] + rng.normal(loc=0, scale=1.0)
    assert model.predict(x4, 1) == y[7]


def test_knn_predict_dataset_2d_and_k_3():
    X = np.array([[0, 0], [0, 1], [1, 1], [3, 1], [3, 3], [4, 3]])
    y = np.array([0, 0, 0, 1, 1, 1])
    x1 = np.array([2, 0])
    x2 = np.array([2, 1])
    x3 = np.array([2, 2])
    model = KNNClassifier(X, y)
    assert model.predict(x1, 3) == 0
    assert model.predict(x2, 3) == 0
    assert model.predict(x3, 3) == 1


def test_knn_predict_dataset_2d_and_k_4():
    X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    y = np.array([1, 0, 0, 1])
    model = KNNClassifier(X, y)
    x0 = np.array([0.2, 0.2])
    x1 = np.array([0.2, -0.2])
    x2 = np.array([-0.2, 0.2])
    x3 = np.array([-0.2, -0.2])
    assert model.predict(x0, 4) == 1
    assert model.predict(x1, 4) == 0
    assert model.predict(x2, 4) == 0
    assert model.predict(x3, 4) == 1


def test_knn_predict_dataset_2d_and_k_5():
    X = np.array([[1, 1], [1, -1], [0, 0.1], [-1, 1], [-1, -1]])
    y = np.array([1, 0, 2, 0, 1])
    model = KNNClassifier(X, y)
    x0 = np.array([0.2, 0.2])
    x1 = np.array([0.2, -0.2])
    x2 = np.array([-0.2, 0.2])
    x3 = np.array([-0.2, -0.2])
    assert model.predict(x0, 5) == 1
    assert model.predict(x1, 5) == 0
    assert model.predict(x2, 5) == 0
    assert model.predict(x3, 5) == 1


def test_knn_predict_dataset_2d_and_k_5():
    X = np.array([[1, 0], [1, 1], [1, -1], [0, 0.1], [-1, 1], [-1, -1]])
    y = np.array([2, 1, 0, 2, 0, 1])
    model = KNNClassifier(X, y)
    x0 = np.array([0.2, 0.2])
    x1 = np.array([0.2, -0.2])
    x2 = np.array([-0.2, 0.2])
    x3 = np.array([-0.2, -0.2])
    assert model.predict(x0, 5) == 2
    assert model.predict(x1, 5) == 2
    assert model.predict(x2, 5) == 2
    assert model.predict(x3, 5) == 2
