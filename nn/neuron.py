"""
Neuron (Perceptron) - The fundamental unit of neural networks.

This is NOT a layer - it's a single computational unit.
A Dense layer is composed of multiple neurons working together.
"""

import numpy as np
from numpy.typing import NDArray

from .activations import ActivationFunction, Linear


class Neuron:
    """
    A single neuron (perceptron) - the fundamental building block.

    Architecture:
        ┌───────────────────────────────────────────┐
        │              Single Neuron                │
        │                                           │
        │   x1 ──w1──┐                              │
        │            │                              │
        │   x2 ──w2──┼──► Σ + b ──► f(z) ──► output │
        │            │                              │
        │   x3 ──w3──┘                              │
        │                                           │
        └───────────────────────────────────────────┘

    Math:
        z = x1*w1 + x2*w2 + ... + xn*wn + b  (weighted sum)
        y = f(z)                              (activation)

    Attributes:
        weights: Array of shape (input_size,) - one weight per input
        bias: Single float value
        activation: Activation function applied to the weighted sum
        grad_weights: Gradient of loss w.r.t. weights
        grad_bias: Gradient of loss w.r.t. bias
    """
    weights: NDArray
    bias: float
    activation: ActivationFunction
    grad_weights: NDArray | None
    grad_bias: float | None
    _input: NDArray | None

    def __init__(self, input_size: int, activation: ActivationFunction | None = None):
        """
        Initialize a single neuron.

        Args:
            input_size: Number of input connections
            activation: Activation function (default: Linear/identity)
        """
        self.activation = activation or Linear()

        # Xavier initialization for a single neuron
        limit = np.sqrt(6 / (input_size + 1))
        self.weights = np.random.uniform(-limit, limit, (input_size))
        self.bias = 0.0

        # Gradients (computed during backward)
        self.grad_weights = None
        self.grad_bias = None

        # Cache for backward pass
        self._input = None

    def forward(self, x: NDArray) -> NDArray:
        """
        Compute neuron output: y = f(Σ(xi * wi) + b)

        Args:
            x: Input array of shape (batch_size, input_size)

        Returns:
            Output array of shape (batch_size,) - one value per sample

        Math:
            z = x1*w1 + x2*w2 + ... + xn*wn + b  (weighted sum)
            y = activation(z)
        """
        self._input = x

        # Weighted sum: z = x · w + b
        z = x @ self.weights + self.bias

        # Apply activation function
        activated = self.activation.forward(z)
        return activated

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Compute gradients for this neuron.

        Args:
            grad_output: Gradient from next layer, shape (batch_size,)
                        This is ∂L/∂y (gradient of loss w.r.t. our output)

        Returns:
            Gradient w.r.t. input, shape (batch_size, input_size)

        Derivation:
            z = Σ(xi * wi) + b
            y = f(z)

            ∂L/∂z  = ∂L/∂y * ∂y/∂z = grad_output * f'(z)

            ∂L/∂wi = Σ(∂L/∂z * ∂z/∂wi) = Σ(∂L/∂z * xi) over batch
            ∂L/∂b  = Σ(∂L/∂z * ∂z/∂b)  = Σ(∂L/∂z * 1) over batch
            ∂L/∂xi = ∂L/∂z * ∂z/∂xi    = ∂L/∂z * wi
        """
        # Gradient through activation: ∂L/∂z = ∂L/∂y * f'(z)
        grad_z = self.activation.backward(grad_output)

        # Gradient w.r.t. weights: sum over batch
        # self._input: (batch_size, input_size)
        # grad_z: (batch_size,)
        self.grad_weights = self._input.T @ grad_z  # (input_size,)

        # Gradient w.r.t. bias: sum over batch
        self.grad_bias = float(np.sum(grad_z))

        # Gradient w.r.t. input: broadcast weights to each sample
        # grad_z: (batch_size,)
        # weights: (input_size,)
        # Result: (batch_size, input_size)
        grad_input = np.outer(grad_z, self.weights)

        return grad_input

    def zero_grad(self) -> None:
        """Reset gradients to None."""
        self.grad_weights = None
        self.grad_bias = None

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Neuron(input_size={self.weights.shape[0]}, activation={self.activation.name})"
