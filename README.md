This is a neural network implementation in Python using NumPy. With configurable activation functions, loss functions, and optimizers, it allows you to build and train neural networks from scratch. For educational purposes.

# Code Base
nn
 L neuron.py -> y = f(w1*x1 + w2*x2 + w3*x3 + ... + b); f is the activation function, W are the weights, X are the inputs, b is the bias
 L activation.py -> y = f(z); f is the activation function, z is the input to the activation function
 L layers.py -> Join multiple neurons.
 L network.py -> Connect layers of neurons.
 L losses.py -> Compute loss between predicted and true values.
 L optimizers.py -> Update model weights based on gradients.