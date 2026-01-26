"""
Neural Network Layers

Each layer implements:
- forward(x): compute output
- backward(grad_output): compute gradient and update internal gradients

The Dense layer uses Neuron objects internally.
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
        │   ┌──────────┐                                     │
        │   │ Neuron 0 │──► z0 ─┐                            │
        │   └──────────┘        │                            │
        │   ┌──────────┐        │    ┌────────────┐          │
        │   │ Neuron 1 │──► z1 ─┼───►│ Activation │──► output│
        │   └──────────┘        │    └────────────┘          │
        │   ┌──────────┐        │                            │
        │   │ Neuron 2 │──► z2 ─┘                            │
        │   └──────────┘                                     │
        │                                                    │
        └────────────────────────────────────────────────────┘

    Each neuron computes: z_i = Σ(x_j * w_ji) + b_i
    Then activation is applied to all outputs: y = f([z0, z1, z2, ...])

    Parameters:
        input_size: Number of input features
        output_size: Number of neurons (output features)
        activation: Activation function (default: Linear/identity)
    """
    input_size: int
    output_size: int
    neurons: list[Neuron]
    activation: ActivationFunction
    _linear_output: NDArray | None
    _weights: NDArray       # Shared weight matrix that neurons reference
    _bias: NDArray          # Shared bias array that neurons reference
    _grad_weights: NDArray | None
    _grad_bias: NDArray | None

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: ActivationFunction = Linear()
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        # Create neurons - each neuron has input_size weights and 1 bias
        self.neurons = [Neuron(input_size) for _ in range(output_size)]

        # Create shared weight/bias arrays for optimizer compatibility
        # These are the "source of truth" - neurons will sync from these
        self._weights = np.column_stack([n.weights for n in self.neurons])
        self._bias = np.array([[n.bias for n in self.neurons]])

        # Gradient arrays
        self._grad_weights = None
        self._grad_bias = None

        # Cache for backward pass
        self._linear_output = None

    def _sync_neurons_from_arrays(self) -> None:
        """Sync neuron weights/biases from the shared arrays."""
        for i, neuron in enumerate(self.neurons):
            neuron.weights = self._weights[:, i].copy()
            neuron.bias = float(self._bias[0, i])

    def _sync_arrays_from_neurons(self) -> None:
        """Sync shared arrays from neuron weights/biases."""
        for i, neuron in enumerate(self.neurons):
            self._weights[:, i] = neuron.weights
            self._bias[0, i] = neuron.bias

    def _sync_gradients_from_neurons(self) -> None:
        """Collect gradients from neurons into shared arrays."""
        if self.neurons[0].grad_weights is not None:
            if self._grad_weights is None:
                self._grad_weights = np.zeros_like(self._weights)
            for i, neuron in enumerate(self.neurons):
                self._grad_weights[:, i] = neuron.grad_weights  # type: ignore

        if self.neurons[0].grad_bias is not None:
            if self._grad_bias is None:
                self._grad_bias = np.zeros_like(self._bias)
            for i, neuron in enumerate(self.neurons):
                self._grad_bias[0, i] = neuron.grad_bias  # type: ignore

    def forward(self, x: NDArray) -> NDArray:
        """
        Forward pass: compute each neuron's output, then apply activation.

        Each neuron computes: z_i = x @ w_i + b_i
        Then: Y = activation([z_0, z_1, ..., z_n])

        Args:
            x: Input array of shape (batch_size, input_size)

        Returns:
            Output array of shape (batch_size, output_size)
        """
        # Sync neurons from shared arrays (in case optimizer updated them)
        self._sync_neurons_from_arrays()

        # Compute each neuron's output
        # Each neuron.forward(x) returns shape (batch_size,)
        neuron_outputs = [neuron.forward(x) for neuron in self.neurons]

        # Stack neuron outputs: -> (batch_size, output_size)
        self._linear_output = np.column_stack(neuron_outputs)

        # Apply activation function
        return self.activation.forward(self._linear_output)

    def backward(self, grad_output: NDArray) -> NDArray:
        """
        Backward pass: propagate gradients through activation, then through each neuron.

        Args:
            grad_output: Gradient from next layer, shape (batch_size, output_size)

        Returns:
            Gradient w.r.t. input, shape (batch_size, input_size)
        """
        # Gradient through activation: ∂L/∂Z = ∂L/∂Y * f'(Z)
        grad_linear = self.activation.backward(grad_output)

        # Propagate gradient to each neuron and accumulate input gradients
        grad_input = np.zeros((grad_output.shape[0], self.input_size))

        for i, neuron in enumerate(self.neurons):
            # Get gradient for this neuron: shape (batch_size,)
            grad_neuron = grad_linear[:, i]

            # Neuron backward returns gradient w.r.t. input
            # and updates neuron.grad_weights and neuron.grad_bias
            grad_input += neuron.backward(grad_neuron)

        # Sync gradients from neurons to shared arrays
        self._sync_gradients_from_neurons()

        return grad_input

    def get_neuron(self, index: int) -> Neuron:
        """
        Get a specific neuron by index.

        Args:
            index: Neuron index (0 to output_size-1)

        Returns:
            The Neuron object
        """
        return self.neurons[index]

    @property
    def weights(self) -> NDArray:
        """Get weights matrix (input_size, output_size)."""
        return self._weights

    @property
    def bias(self) -> NDArray:
        """Get bias array (1, output_size)."""
        return self._bias

    @property
    def grad_weights(self) -> NDArray | None:
        """Get weight gradients matrix."""
        return self._grad_weights

    @property
    def grad_bias(self) -> NDArray | None:
        """Get bias gradients array."""
        return self._grad_bias

    @property
    def parameters(self) -> list[NDArray]:
        """Return weights and bias for optimizer (these are modified in-place)."""
        return [self._weights, self._bias]

    @property
    def gradients(self) -> list[NDArray | None]:
        """Return gradients for optimizer."""
        return [self._grad_weights, self._grad_bias]

    def __repr__(self) -> str:
        return f"Dense({self.input_size}, {self.output_size}, activation={self.activation.name}, neurons={len(self.neurons)})"
