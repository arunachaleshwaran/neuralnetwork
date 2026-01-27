"""
Neural Network Layers

Each layer implements:
- forward(x): compute output
- backward(grad_output): compute gradient and update internal gradients

The Dense layer uses Neuron objects internally.
Each Neuron handles its own activation function.
See neuron.py for the single neuron implementation.
"""

from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

from .activations import ActivationFunction, Linear
from .neuron import Neuron


class Layer(ABC):
    """Base class for all layers."""
    trainable: bool

    def __init__(self):
        self.trainable = True

    @abstractmethod
    def forward(self, x: NDArray) -> NDArray:
        """Forward pass."""
        pass

    @abstractmethod
    def backward(self, grad_output: NDArray) -> NDArray:
        """Backward pass. Returns gradient w.r.t. input."""
        pass

    def __call__(self, x: NDArray) -> NDArray:
        return self.forward(x)

    def log_state(self) -> None:
        """Log internal state for debugging."""
        pass

    @property
    def parameters(self) -> list[NDArray]:
        """Return list of trainable parameters."""
        return []

    @property
    def gradients(self) -> list[NDArray | None]:
        """Return list of gradients for parameters."""
        return []


class Dense(Layer):
    """
    Fully connected (dense) layer - composed of multiple Neuron objects.

    Architecture:
        ┌────────────────────────────────────────────────────┐
        │                Dense Layer                         │
        │                                                    │
        │   ┌──────────────────────┐                         │
        │   │ Neuron 0 (with f(z)) │──► y0 ─┐                │
        │   └──────────────────────┘        │                │
        │   ┌──────────────────────┐        │                │
        │   │ Neuron 1 (with f(z)) │──► y1 ─┼───► output     │
        │   └──────────────────────┘        │                │
        │   ┌──────────────────────┐        │                │
        │   │ Neuron 2 (with f(z)) │──► y2 ─┘                │
        │   └──────────────────────┘                         │
        │                                                    │
        └────────────────────────────────────────────────────┘

    Each neuron computes: y_i = f(Σ(x_j * w_ji) + b_i)
    The activation function is handled by each neuron internally.

    Parameters:
        input_size: Number of input features
        output_size: Number of neurons (output features)
        activation: Activation function for all neurons (default: Linear/identity)
    """
    input_size: int
    output_size: int
    neurons: list[Neuron]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction = Linear()
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Create neurons - each neuron has its own activation function
        # We create new instances of the same activation type for each neuron
        self.neurons = [
            Neuron(input_size, activation=type(activation)())
            for _ in range(output_size)
        ]

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward pass: each neuron computes its output (including activation).

        Args:
            x: Input array of shape (batch_size, input_size)

        Returns:
            Output array of shape (batch_size, output_size)
        """
        # Each neuron computes: y = f(x @ w + b)
        # neuron.forward(x) returns shape (batch_size,)
        neuron_outputs = [neuron.forward(x) for neuron in self.neurons]

        # Stack outputs: -> (batch_size, output_size)
        return np.column_stack(neuron_outputs)

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Backward pass: propagate gradients through each neuron.

        Each neuron handles its own activation gradient internally.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_size)

        Returns:
            Gradient w.r.t. input, shape (batch_size, input_size)
        """
        # Propagate gradient to each neuron and accumulate input gradients
        grad_input = np.zeros((grad_output.shape[0], self.input_size))

        for i, neuron in enumerate(self.neurons):
            # Get gradient for this neuron: shape (batch_size,)
            grad_neuron = grad_output[:, i]

            # Neuron backward handles activation gradient internally
            # and updates neuron.grad_weights and neuron.grad_bias
            grad_input += neuron.backward(grad_neuron)

        return grad_input
    def log_state(self) -> None:
        """Log internal state for debugging."""
        for i, neuron in enumerate(self.neurons):
            print(f"  Neuron {i + 1}: weights={neuron.weights}, bias={neuron.bias:.6f}")
    @property
    def parameters(self) -> list[NDArray]:
        """Return neuron weights and biases for optimizer."""
        params: list[NDArray] = []
        for neuron in self.neurons:
            params.append(neuron.weights)
        for neuron in self.neurons:
            params.append(np.array([neuron.bias]))
        return params

    @property
    def gradients(self) -> list[NDArray | None]:
        """Return gradients from neurons for optimizer."""
        grads: list[NDArray | None] = []
        for neuron in self.neurons:
            grads.append(neuron.grad_weights)
        for neuron in self.neurons:
            grads.append(np.array([neuron.grad_bias]) if neuron.grad_bias is not None else None)
        return grads

    def __repr__(self) -> str:
        return f"Dense({self.input_size}, {self.output_size}, activation={self.neurons[0].activation.name}, neurons={len(self.neurons)})"
