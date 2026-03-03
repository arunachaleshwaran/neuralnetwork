"""
Activation Functions for Neural Networks

Each activation function implements:
- forward(x): compute output
- backward(grad_output): compute gradient for backpropagation
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class ActivationFunction(ABC):
    """Base class for all activation functions."""
    _cache: NDArray | None
    
    def __init__(self):
        self._cache: NDArray | None = None

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        """Compute f(x)."""
        pass

    @abstractmethod
    def backward(self, grad_output: NDArray) -> NDArray:
        """Compute gradient using cached input/output."""
        pass

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    @property
    def name(self) -> str:
        return self.__class__.__name__


class ReLU(ActivationFunction):
    """
    Rectified Linear Unit: f(x) = max(0, x)

    Gradient: 1 if x > 0, else 0
    """

    def forward(self, x: NDArray) -> NDArray:
        self._cache = x
        return np.maximum(0, x)

    def backward(self, grad_output: NDArray) -> NDArray:
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        return grad_output * (self._cache > 0).astype(float)


class Sigmoid(ActivationFunction):
    """
    Sigmoid: f(x) = 1 / (1 + exp(-x))

    Gradient: f(x) * (1 - f(x))
    """

    def forward(self, x: NDArray) -> NDArray:
        # Clip to prevent overflow
        x = np.clip(x, -500, 500)
        output = 1 / (1 + np.exp(-x))
        self._cache = output
        return output

    def backward(self, grad_output: NDArray) -> NDArray:
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        return grad_output * self._cache * (1 - self._cache)


class Tanh(ActivationFunction):
    """
    Hyperbolic tangent: f(x) = tanh(x)

    Gradient: 1 - tanh(x)^2
    """

    def forward(self, x: NDArray) -> NDArray:
        output = np.tanh(x)
        self._cache = output
        return output

    def backward(self, grad_output: NDArray) -> NDArray:
        if self._cache is None:
            raise ValueError("Must call forward() before backward()")
        return grad_output * (1 - self._cache ** 2)


class Softmax(ActivationFunction):
    """
    Softmax: f(x_i) = exp(x_i) / sum(exp(x_j))

    Used for multi-class classification output layer.
    Note: Usually combined with CrossEntropy loss for stable gradients.
    """

    def forward(self, x: NDArray) -> NDArray:
        # Subtract max for numerical stability
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self._cache = output
        return output

    def backward(self, grad_output: NDArray) -> NDArray:
        # Simplified gradient when used with cross-entropy
        # Full Jacobian computation is complex; this works for most cases
        return grad_output


class Linear(ActivationFunction):
    """Identity activation: f(x) = x"""

    def forward(self, x: NDArray) -> NDArray:
        self._cache = x
        return x

    def backward(self, grad_output: NDArray) -> NDArray:
        return grad_output
