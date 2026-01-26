"""
Optimizers for Neural Networks

Optimizers update the network parameters based on computed gradients.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        """Update parameters using their gradients."""
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Update rule:
        v = momentum * v - learning_rate * gradient
        param = param + v

    Parameters:
        learning_rate: Step size for updates
        momentum: Momentum factor (0 = no momentum)
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self._velocities: list[NDArray] | None = None

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        # Initialize velocities on first call
        if self._velocities is None:
            self._velocities = [np.zeros_like(p) for p in params]

        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is None:
                continue

            self._velocities[i] = self.momentum * self._velocities[i] - self.learning_rate * grad
            param += self._velocities[i]


class Adam(Optimizer):
    """
    Adam optimizer (Adaptive Moment Estimation)

    Combines momentum with adaptive learning rates.

    Parameters:
        learning_rate: Step size (default: 0.001)
        beta1: Exponential decay rate for first moment (default: 0.9)
        beta2: Exponential decay rate for second moment (default: 0.999)
        epsilon: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self._t = 0  # Time step
        self._m: list[NDArray] | None = None  # First moment
        self._v: list[NDArray] | None = None  # Second moment

    def update(self, params: list[NDArray], grads: list[NDArray]) -> None:
        # Initialize moments on first call
        if self._m is None:
            self._m = [np.zeros_like(p) for p in params]
            self._v = [np.zeros_like(p) for p in params]

        self._t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            if grad is None:
                continue

            # Update biased first moment estimate
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (grad ** 2)

            # Bias correction
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)

            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
