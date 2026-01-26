"""
Neuron (Perceptron) - The fundamental unit of neural networks.

This is NOT a layer - it's a single computational unit.
A Dense layer is composed of multiple neurons working together.
"""

import numpy as np
from numpy.typing import NDArray


class Neuron:
    """
    A single neuron (perceptron) - the fundamental building block.

    Architecture:
        ┌─────────────────────────────────────┐
        │           Single Neuron             │
        │                                     │
        │   x1 ──w1──┐                        │
        │            │                        │
        │   x2 ──w2──┼──► Σ + b ──► z (output)│
        │            │                        │
        │   x3 ──w3──┘                        │
        │                                     │
        └─────────────────────────────────────┘

    Note: Activation is typically applied at the layer level,
    not per-neuron, for efficiency. This neuron computes only
    the linear transformation: z = Σ(xi * wi) + b

    Attributes:
        weights: Array of shape (input_size,) - one weight per input
        bias: Single float value
        grad_weights: Gradient of loss w.r.t. weights
        grad_bias: Gradient of loss w.r.t. bias
    """
    input_size: int
    weights: NDArray
    bias: float
    grad_weights: NDArray | None
    grad_bias: float | None
    _input: NDArray | None

    def __init__(self, input_size: int):
        """
        Initialize a single neuron.

        Args:
            input_size: Number of input connections
        """
        self.input_size = input_size

        # Xavier initialization for a single neuron
        limit = np.sqrt(6 / (input_size + 1))
        self.weights = np.random.uniform(-limit, limit, (input_size,))
        self.bias = 0.0

        # Gradients (computed during backward)
        self.grad_weights = None
        self.grad_bias = None

        # Cache for backward pass
        self._input = None

    def forward(self, x: NDArray) -> NDArray:
        """
        Compute weighted sum: z = Σ(xi * wi) + b

        Args:
            x: Input array of shape (batch_size, input_size)

        Returns:
            Output array of shape (batch_size,) - one value per sample

        Math:
            For each sample: z = x1*w1 + x2*w2 + ... + xn*wn + b
        """
        self._input = x
        return x @ self.weights + self.bias

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Compute gradients for this neuron.

        Args:
            grad_output: Gradient from next layer, shape (batch_size,)
                        This is ∂L/∂z (gradient of loss w.r.t. our output)

        Returns:
            Gradient w.r.t. input, shape (batch_size, input_size)

        Derivation:
            z = Σ(xi * wi) + b

            ∂L/∂wi = Σ(∂L/∂z * ∂z/∂wi) = Σ(grad_output * xi) over batch
            ∂L/∂b  = Σ(∂L/∂z * ∂z/∂b)  = Σ(grad_output * 1) over batch
            ∂L/∂xi = ∂L/∂z * ∂z/∂xi    = grad_output * wi
        """
        # Gradient w.r.t. weights: sum over batch
        # self._input: (batch_size, input_size)
        # grad_output: (batch_size,)
        self.grad_weights = self._input.T @ grad_output  # (input_size,)

        # Gradient w.r.t. bias: sum over batch
        self.grad_bias = float(np.sum(grad_output))

        # Gradient w.r.t. input: broadcast weights to each sample
        # grad_output: (batch_size,) -> (batch_size, 1)
        # weights: (input_size,)
        # Result: (batch_size, input_size)
        grad_input = np.outer(grad_output, self.weights)

        return grad_input

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def __repr__(self) -> str:
        return f"Neuron(input_size={self.input_size})"
