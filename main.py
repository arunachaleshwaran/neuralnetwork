"""
Neural Network Demo: XOR Problem

The XOR problem is a classic example that demonstrates why we need
hidden layers - it's not linearly separable.

XOR Truth Table:
    Input    | Output
    0, 0     | 0
    0, 1     | 1
    1, 0     | 1
    1, 1     | 0
"""

import numpy as np
from nn import NeuralNetwork, Dense, ReLU, Sigmoid, Tanh, MSE, Adam, SGD
from nn.activations import Linear

def demo_height_prediction():
    """Demo: Predict height from age and weight."""
    print("\n" + "=" * 60)
    print("Demo 1: Height Prediction from Age and Weight")
    print("=" * 60)

    # Sample dataset: [age] -> height
    X = np.array([
        [0],
        [1.38],
        [2.76],
        [4.14],
        [5.52],
        [6.9],
        [8.28],
        [9.66],
        [11.03],
        [12.41],
        [13.79],
        [15.17],
        [16.55],
        [17.93],
        [19.31],
        [20.69],
        [22.07],
        [23.45],
        [24.83],
        [26.21],
        [27.59],
        [28.97],
        [30.34],
        [31.72],
        [33.1],
        [34.48],
        [35.86],
        [37.24],
        [38.62],
        [40.0],
    ], dtype=np.float64)

    y = np.array([
        [50.2],
        [101.3],
        [129.2],
        [135.7],
        [142.3],
        [149.9],
        [156.1],
        [163.1],
        [170.4],
        [175.8],
        [179],
        [183.4],
        [186.1],
        [190],
        [190.4],
        [189.8],
        [190.2],
        [189.6],
        [189.6],
        [189.9],
        [190.3],
        [190.4],
        [189.9],
        [190],
        [189.7],
        [190.4],
        [190.1],
        [189.8],
        [190],
        [190.1],
    ], dtype=np.float64)

    print("\nSample Dataset:")
    print("  Age  | Height")
    for i in range(len(X)):
        print(f"  {X[i][0]:>5.2f} => {y[i][0]:>6.1f}")

    # Normalize data for better training
    X_mean, X_std = X.mean(), X.std()
    y_mean, y_std = y.mean(), y.std()
    X_norm = (X - X_mean) / X_std
    y_norm = (y - y_mean) / y_std

    # Create network - larger capacity to fit the non-linear pattern
    model = NeuralNetwork()
    model.add(Dense(1, 64, activation=ReLU()))
    model.add(Dense(64, 32, activation=ReLU()))
    model.add(Dense(32, 1, activation=Linear()))

    model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))

    print("\nNetwork Architecture:")
    model.summary()

    print("\nTraining...")
    model.fit(X_norm, y_norm, epochs=2000, verbose=True, batch_size=2)

    # Test predictions
    test_ages = np.array([[3], [7], [10], [14], [18], [21], [25], [29], [32], [36], [39]], dtype=np.float64)
    test_norm = (test_ages - X_mean) / X_std
    predictions_norm = model.predict(test_norm)
    predictions = predictions_norm * y_std + y_mean

    print("\nPredictions:")
    print("  Age => Predicted Height")
    for i in range(len(test_ages)):
        print(f"  {test_ages[i][0]:>3.0f} => {predictions[i][0]:>16.2f}")

    # Final loss (on normalized data, close to 0 is good)
    final_loss = model.evaluate(X_norm, y_norm)
    print(f"\nFinal Loss (normalized): {final_loss:.6f}")

    # Also show loss in original scale
    predictions_train = model.predict(X_norm) * y_std + y_mean
    mse_original = np.mean((predictions_train - y) ** 2)
    print(f"Final MSE (original scale): {mse_original:.2f}")


def demo_xor():
    """Train a neural network to learn the XOR function."""
    print("=" * 60)
    print("Neural Network from Scratch - XOR Demo")
    print("=" * 60)

    # XOR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.float64)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=np.float64)

    print("\nXOR Dataset:")
    print("  Input -> Output")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i][0]}")

    # Create the neural network
    # Architecture: 2 inputs -> 4 hidden neurons -> 1 output
    model = NeuralNetwork()
    model.add(Dense(2, 4, activation=Tanh()))    # Hidden layer
    model.add(Dense(4, 1, activation=Sigmoid())) # Output layer

    # Compile with loss function and optimizer
    model.compile(
        loss=MSE(),
        optimizer=Adam(learning_rate=0.1)
    )

    print("\nNetwork Architecture:")
    model.summary()

    # Train the network
    print("\nTraining...")
    history = model.fit(X, y, epochs=1000, verbose=True)

    # Test the network
    print("\nPredictions after training:")
    print("  Input    | Target | Predicted | Rounded")
    print("  " + "-" * 40)

    predictions = model.predict(X)
    for i in range(len(X)):
        pred = predictions[i][0]
        rounded = round(pred)
        correct = "✓" if rounded == y[i][0] else "✗"
        print(f"  {X[i]} | {y[i][0]:.0f}      | {pred:.4f}    | {rounded} {correct}")

    # Final loss
    final_loss = model.evaluate(X, y)
    print(f"\nFinal Loss: {final_loss:.6f}")

    return history


def demo_binary_classification():
    """Demo: Simple binary classification with a larger dataset."""
    print("\n" + "=" * 60)
    print("Demo 2: Binary Classification (Circle Dataset)")
    print("=" * 60)

    # Generate circular dataset
    np.random.seed(42)
    n_samples = 200

    # Inner circle (class 0)
    r1 = np.random.uniform(0, 0.5, n_samples // 2)
    theta1 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X1 = np.column_stack([r1 * np.cos(theta1), r1 * np.sin(theta1)])
    y1 = np.zeros((n_samples // 2, 1))

    # Outer ring (class 1)
    r2 = np.random.uniform(0.7, 1.0, n_samples // 2)
    theta2 = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    X2 = np.column_stack([r2 * np.cos(theta2), r2 * np.sin(theta2)])
    y2 = np.ones((n_samples // 2, 1))

    X = np.vstack([X1, X2])
    y = np.vstack([y1, y2])

    # Shuffle
    indices = np.random.permutation(n_samples)
    X, y = X[indices], y[indices]

    # Split into train/test
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nDataset: {n_samples} samples (inner circle vs outer ring)")
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Create and train network
    model = NeuralNetwork()
    model.add(Dense(2, 8, activation=ReLU()))
    model.add(Dense(8, 4, activation=ReLU()))
    model.add(Dense(4, 1, activation=Sigmoid()))

    model.compile(loss=MSE(), optimizer=Adam(learning_rate=0.01))

    print("\nNetwork Architecture:")
    model.summary()

    print("\nTraining...")
    model.fit(X_train, y_train, epochs=500, batch_size=32, verbose=True)

    # Evaluate
    train_loss = model.evaluate(X_train, y_train)
    test_loss = model.evaluate(X_test, y_test)

    # Calculate accuracy
    train_preds = (model.predict(X_train) > 0.5).astype(float)
    test_preds = (model.predict(X_test) > 0.5).astype(float)
    train_acc = np.mean(train_preds == y_train) * 100
    test_acc = np.mean(test_preds == y_test) * 100

    print(f"\nResults:")
    print(f"  Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.1f}%")
    print(f"  Test Loss:  {test_loss:.4f}, Accuracy: {test_acc:.1f}%")


def main():
    """Run all demos."""
    # Set random seed for reproducibility
    np.random.seed(42)
    demo_height_prediction()
    # Demo 1: XOR problem
    # demo_xor()

    # Demo 2: Binary classification
    # demo_binary_classification()

    print("\n" + "=" * 60)
    print("Done! Neural network built from scratch using only NumPy.")
    print("=" * 60)


if __name__ == "__main__":
    main()
