import numpy as np
import pytest
from src.knn import KNN
from sklearn.datasets import make_blobs


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


def test_knn_constructor(dataset_2d):
    X, y = dataset_2d
    model = KNN(X, y)
    assert (model.X_train == X).all()
    assert len(model.X_train) == 9
    assert (model.y_train == y).all()
    assert len(model.y_train) == 9


def test_knn_constructor_not_ndarray(dataset_2d):
    X_list = [[2, 3, 1, 4], [-1, 2, 3, 5]]
    y_list = [[10, 5]]
    X_np, y_np = dataset_2d
    with pytest.raises(AssertionError):
        _ = KNN(X_list, y_np)
    with pytest.raises(AssertionError):
        _ = KNN(X_np, y_list)


def test_knn_distance():
    arr1 = np.array([[2.2, 5], [-1.0, 2]])
    arr2 = np.array([[4, 1.1]])
    dist = KNN.calculate_distance(arr1, arr2)
    assert len(dist) == 2
    assert dist[0] == pytest.approx(4.2953, 0.0001)
    assert dist[1] == pytest.approx(5.0803, 0.0001)


def test_knn_get_smallest_distance_idx_different_values():
    dist = np.array([10.3, 7.77, 3.1, 4, 5.5, 8.99, 5.4])
    idx = KNN.get_smallest_distance_idx(dist, 1)
    assert len(idx) == 1
    assert idx[0] == 2  # indice 2 smallest value
    idx = KNN.get_smallest_distance_idx(dist, 4)
    assert len(idx) == 4
    assert idx[0] == 2
    assert idx[1] == 3
    assert idx[2] == 6
    assert idx[3] == 4


def test_knn_get_smallest_distance_idx_repeated_values():
    dist = np.array([4, 3.1, 3.7, 3.1, 4, 5.5, 8.99, 5.4])
    idx = KNN.get_smallest_distance_idx(dist, 1)
    assert len(idx) == 1
    assert idx[0] == 1  # indice 1 smallest value
    idx = KNN.get_smallest_distance_idx(dist, 5)
    assert len(idx) == 5
    assert idx[0] == 1
    assert idx[1] == 3
    assert idx[2] == 2
    assert idx[3] == 0
    assert idx[4] == 4
