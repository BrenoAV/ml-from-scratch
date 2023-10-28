# -*- coding: utf-8 -*-
"""
KNN implementation from scratch
Author: BrenoAV
"""
import sys
from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


class KNN:
    """
    A interface with key elements of KNN algorithm
    """

    def __init__(self, X_train: NDArray, y_train: NDArray) -> None:
        assert len(X_train) == len(
            y_train
        ), f"The length of X_train ({len(X_train)}) and y_train ({len(y_train)}) is different"
        assert isinstance(
            X_train, np.ndarray
        ), f"X_train of type {type(X_train)} is not a numpy array"
        assert isinstance(
            y_train, np.ndarray
        ), f"y_train of type {type(y_train)} is not a numpy array"
        self._X_train = X_train
        self._y_train = y_train

    @property
    def X_train(self) -> NDArray:
        return self._X_train

    @property
    def y_train(self) -> NDArray:
        return self._y_train

    @staticmethod
    def calculate_distance(X: NDArray, x: NDArray) -> NDArray:
        """
        Calculate the L1 distances between a array with shape (1, D) and a vectors
        with the shape (N, D)

        Args:
            X (NDArray): An array of shape (N, D) containing N vectors with D features
            x (NDArray): A single array with shape (1, D) representing a sample

        Returns:
            NDArray: A numpy array (N, ) with the distance between the vectors of X and x
        """
        return np.linalg.norm(np.subtract(X, x), axis=1).flatten()

    @staticmethod
    def get_smallest_distance_idx(distances: NDArray, k: int) -> list[int]:
        """Get a vector of size `k` with the indices contains the smallest distances

        Args:
            distances (NDArray): An array of shape (N, ) represeting the distances
            k (int): Number of elements to be returned

        Returns:
            list[int]: A list with the indices of size `k` that contains
                       the smallest distances
        """
        return list(distances.argsort()[:k])

    def __repr__(self) -> str:
        return f"--- KNN Model ---\nTraining data size: {len(self._X_train)}"


class KNNClassifier(KNN):
    """
    KNN implementation for classification problems
    """

    def __init__(self, X_train: NDArray, y_train: NDArray) -> None:
        super().__init__(X_train, y_train)

    def predict(self, x: NDArray, k: int = 1) -> int:
        """Predict a new entry using the training data and k-neighbors to estimate

        Args:
            x (NDArray): An array 1D with the new entry to be predicted
            k (int, optional): Number of neighbors to be consired. Defaults to 1.

        Returns:
            int: return a int which represent a class from the y_train data (majority vote)
        """
        distances = self.calculate_distance(self.X_train, x)
        smallest_idx = self.get_smallest_distance_idx(distances, k)
        poll = defaultdict(int)
        min_dist = defaultdict(lambda: sys.float_info.max)
        for i, dist in zip(smallest_idx, distances[smallest_idx]):
            poll[self.y_train[i]] += 1
            min_dist[self.y_train[i]] = min(dist, min_dist[self.y_train[i]])
        # Get the highest number of equal neighbors
        max_value = max(poll.values())
        # Get he neighbors with the highest number to untie
        max_idx_values = [k for k, v in poll.items() if v == max_value]
        untie_dict = {key: min_dist[key] for key in max_idx_values}
        # The untie approach is take the element with the minimum distance
        majority = min(untie_dict, key=untie_dict.get)
        return majority


class KNNRegressor(KNN):
    """
    KNN implementation for regression problems
    """

    def __init__(self, X_train: NDArray, y_train: NDArray) -> None:
        super().__init__(X_train, y_train)

    def predict(self, x: NDArray, k: int = 1) -> float:
        """Predict a new entry using the training data and k-neighbors to estimate

        Args:
            x (NDArray): An array 1D with the new entry to be predicted
            k (int, optional): Number of neighbors to be consired. Defaults to 1.

        Returns:
            int: return a float which is the meaning of the k-neighbors
        """
        distances = self.calculate_distance(self.X_train, x)
        smallest_idx = self.get_smallest_distance_idx(distances, k)
        values = self.y_train[smallest_idx]
        mean_values = values.mean()
        return mean_values
