"""
Neural Network Class

The main class that ties together layers, loss functions, and optimizers
to create a trainable neural network.
"""

import numpy as np
from numpy.typing import NDArray

from .layers import Layer
from .losses import Loss
from .optimizers import Optimizer


class NeuralNetwork:
    """
    A sequential neural network.

    Example:
        >>> from nn import NeuralNetwork, Dense, ReLU, Sigmoid, MSE, Adam
        >>>
        >>> # Create network
        >>> model = NeuralNetwork()
        >>> model.add(Dense(2, 4, activation=ReLU()))
        >>> model.add(Dense(4, 1, activation=Sigmoid()))
        >>>
        >>> # Compile with loss and optimizer
        >>> model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))
        >>>
        >>> # Train
        >>> history = model.fit(X_train, y_train, epochs=100, batch_size=32)
        >>>
        >>> # Predict
        >>> predictions = model.predict(X_test)
    """
    layers: list[Layer]
    loss: Loss | None
    optimizer: Optimizer | None

    def __init__(self):
        self.layers = []
        self.loss = None
        self.optimizer = None

    def add(self, layer: Layer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    def compile(self, loss: Loss, optimizer: Optimizer) -> None:
        """Configure the network for training."""
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, x: NDArray) -> NDArray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: NDArray) -> None:
        """Backward pass through all layers (in reverse)."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _get_all_parameters(self) -> list[NDArray]:
        """Collect all trainable parameters from all layers."""
        params = []
        for layer in self.layers:
            if layer.trainable:
                params.extend(layer.parameters)
        return params

    def _get_all_gradients(self) -> list[NDArray]:
        """Collect all gradients from all layers."""
        grads = []
        for layer in self.layers:
            if layer.trainable:
                grads.extend(layer.gradients)
        return grads

    def _train_step(self, x_batch: NDArray, y_batch: NDArray) -> np.floating:
        """Perform one training step on a batch."""
        if self.loss is None:
            raise ValueError("Loss function must be set before training")
        if self.optimizer is None:
            raise ValueError("Optimizer must be set before training")
        # Forward pass
        predictions = self.forward(x_batch)

        # Compute loss
        loss_value = self.loss.forward(predictions, y_batch)

        # Backward pass
        grad = self.loss.backward()
        self.backward(grad)

        # Update parameters
        params = self._get_all_parameters()
        grads = self._get_all_gradients()
        self.optimizer.update(params, grads)

        return loss_value

    def fit(
        self,
        x: NDArray,
        y: NDArray,
        epochs: int = 100,
        batch_size: int | None = None,
        verbose: bool = True
    ) -> dict:
        """
        Train the network.

        Args:
            x: Training data, shape (n_samples, n_features)
            y: Target values, shape (n_samples, n_outputs)
            epochs: Number of training epochs
            batch_size: Batch size (None = use all data)
            verbose: Print training progress

        Returns:
            Dictionary with training history (loss per epoch)
        """
        if self.loss is None or self.optimizer is None:
            raise ValueError("Call compile() before fit()")

        n_samples = x.shape[0]
        if batch_size is None:
            batch_size = n_samples

        history = {"loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            # Mini-batch training
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                batch_loss = self._train_step(x_batch, y_batch)
                epoch_loss += batch_loss
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            history["loss"].append(avg_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")
                self.log_state()

        return history

    def predict(self, x: NDArray) -> NDArray:
        """Make predictions."""
        return self.forward(x)

    def evaluate(self, x: NDArray, y: NDArray) -> np.floating:
        """Evaluate loss on given data."""
        if self.loss is None:
            raise ValueError("Loss function must be set before evaluation")
        predictions = self.forward(x)
        return self.loss.forward(predictions, y)

    def summary(self) -> None:
        """Print a summary of the network architecture."""
        print("=" * 50)
        print("Neural Network Summary")
        print("=" * 50)
        total_params = 0
        for i, layer in enumerate(self.layers):
            params = sum(p.size for p in layer.parameters)
            total_params += params
            print(f"Layer {i + 1}: {layer} - {params:,} parameters")
        print("=" * 50)
        print(f"Total parameters: {total_params:,}")
        print("=" * 50)

    def log_state(self) -> None:
        """Log current state of the network with all neuron weights and biases."""
        print("-" * 50)
        print("Network State")
        print("-" * 50)
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1}: {layer}")
            layer.log_state()
        print("-" * 50)

    def __repr__(self) -> str:
        return f"NeuralNetwork(layers={len(self.layers)})"
