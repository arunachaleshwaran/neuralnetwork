"""
Loss Functions for Neural Networks

Each loss function implements:
- forward(y_pred, y_true): compute loss value
- backward(): compute gradient w.r.t. predictions
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Loss(ABC):
    """Base class for all loss functions."""

    def __init__(self):
        self._y_pred: NDArray | None = None
        self._y_true: NDArray | None = None

    @abstractmethod
    def forward(self, y_pred: NDArray, y_true: NDArray) -> np.floating:
        """Compute loss value."""

    @abstractmethod
    def backward(self) -> NDArray:
        """Compute gradient w.r.t. predictions."""

    def __call__(self, y_pred: NDArray, y_true: NDArray) -> np.floating:
        return self.forward(y_pred, y_true)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class MSE(Loss):
    """
    Mean Squared Error Loss

    L = (1/n) * sum((y_pred - y_true)^2)

    Used for regression problems.
    """
    
    def forward(self, y_pred: NDArray, y_true: NDArray) -> np.floating:
        self._y_pred = y_pred
        self._y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> NDArray:
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        n = self._y_pred.shape[0]
        return 2 * (self._y_pred - self._y_true) / n


class CrossEntropy(Loss):
    """
    Cross-Entropy Loss (for classification)

    L = -sum(y_true * log(y_pred))

    Used with softmax output for multi-class classification.
    For binary classification, use with sigmoid output.
    """

    def forward(self, y_pred: NDArray, y_true: NDArray) -> np.floating:
        self._y_pred = y_pred
        self._y_true = y_true

        # Clip to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        # If y_true is one-hot encoded
        if y_true.ndim == 2:
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
        else:
            # y_true is class indices
            batch_size = y_pred.shape[0]
            loss = -np.sum(np.log(y_pred_clipped[np.arange(batch_size), y_true])) / batch_size

        return loss

    def backward(self) -> NDArray:
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        # Simplified gradient for softmax + cross-entropy
        # dL/dy_pred = y_pred - y_true (when used with softmax)
        if self._y_true.ndim == 2:
            return (self._y_pred - self._y_true) / self._y_true.shape[0]
        else:
            # Convert class indices to one-hot
            batch_size = self._y_pred.shape[0]
            one_hot = np.zeros_like(self._y_pred)
            one_hot[np.arange(batch_size), self._y_true] = 1
            return (self._y_pred - one_hot) / batch_size


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy Loss

    L = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

    Used for binary classification with sigmoid output.
    """

    def forward(self, y_pred: NDArray, y_true: NDArray) -> np.floating:
        self._y_pred = y_pred
        self._y_true = y_true

        # Clip to prevent log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)

        loss = -np.mean(
            y_true * np.log(y_pred_clipped) +
            (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return np.floating(loss)

    def backward(self) -> NDArray:
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        # Clip to prevent division by zero
        y_pred_clipped = np.clip(self._y_pred, 1e-15, 1 - 1e-15)
        n = self._y_pred.shape[0]
        return (-(self._y_true / y_pred_clipped) + (1 - self._y_true) / (1 - y_pred_clipped)) / n
