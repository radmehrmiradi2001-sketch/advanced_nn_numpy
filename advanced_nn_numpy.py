

import argparse
import os
import sys
import random
from typing import Callable, List, Tuple, Dict, Any

import numpy as np


# -----------------------------------------------------------------------------
# Activation functions and their derivatives
# -----------------------------------------------------------------------------
def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def softmax(x: np.ndarray) -> np.ndarray:
    # Numerically stable softmax
    shift_x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


ACTIVATIONS: Dict[str, Tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
}


# -----------------------------------------------------------------------------
# Loss functions and their derivatives
# -----------------------------------------------------------------------------
class LossFunction:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return self.loss(y_pred, y_true)

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError


class MSELoss(LossFunction):
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (2.0 / y_true.shape[0]) * (y_pred - y_true)


class CrossEntropyLoss(LossFunction):
    def loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        # Avoid log(0)
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        # Derivative of cross entropy with softmax output simplifies to y_pred - y_true
        return (y_pred - y_true) / y_true.shape[0]


LOSSES: Dict[str, LossFunction] = {
    "mse": MSELoss(),
    "cross_entropy": CrossEntropyLoss(),
}


# -----------------------------------------------------------------------------
# Weight initialization methods
# -----------------------------------------------------------------------------
def initialize_weights(n_in: int, n_out: int, method: str = "he") -> np.ndarray:
    
    if method == "he":
        return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
    elif method == "xavier":
        return np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
    else:  # default normal
        return np.random.randn(n_in, n_out) * 0.01


# -----------------------------------------------------------------------------
# Neural network class implementation
# -----------------------------------------------------------------------------
class NeuralNetwork:
    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        loss: str = "cross_entropy",
        weight_init: str = "he",
        learning_rate: float = 0.01,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        l2_lambda: float = 0.0,
        dropout_prob: float = 0.0,
    ):
        assert len(layer_sizes) - 1 == len(activations), "Number of activations must match number of layers"
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.loss_fn = LOSSES[loss]
        self.weight_init = weight_init
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.l2_lambda = l2_lambda
        self.dropout_prob = dropout_prob
        self.use_dropout = dropout_prob > 0.0

        # Initialize parameters
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(layer_sizes) - 1):
            n_in, n_out = layer_sizes[i], layer_sizes[i + 1]
            W = initialize_weights(n_in, n_out, method=weight_init)
            b = np.zeros((1, n_out))
            self.weights.append(W)
            self.biases.append(b)

        # Initialize optimizer state
        if optimizer == "momentum":
            self.velocities_W = [np.zeros_like(W) for W in self.weights]
            self.velocities_b = [np.zeros_like(b) for b in self.biases]
        elif optimizer == "adam":
            self.m_W = [np.zeros_like(W) for W in self.weights]
            self.v_W = [np.zeros_like(W) for W in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0

    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        
        activations = [X]
        pre_activations = []
        dropout_masks = []
        A = X
        for i in range(len(self.weights)):
            W = self.weights[i]
            b = self.biases[i]
            Z = np.dot(A, W) + b  # (batch, n_out)
            pre_activations.append(Z)
            # Select activation function for this layer
            act_name = self.activations[i]
            if act_name == "softmax":
                A = softmax(Z)
            elif act_name == "linear":
                A = linear(Z)
            else:
                A = ACTIVATIONS[act_name][0](Z)

            # Dropout on hidden layers (not on output layer)
            if self.use_dropout and training and i < len(self.weights) - 1:
                keep_prob = 1.0 - self.dropout_prob
                mask = (np.random.rand(*A.shape) < keep_prob) / keep_prob
                A *= mask
                dropout_masks.append(mask)
            else:
                dropout_masks.append(None)

            activations.append(A)
        return activations, pre_activations, dropout_masks

    def backward(
        self,
        activations: List[np.ndarray],
        pre_activations: List[np.ndarray],
        y_true: np.ndarray,
        dropout_masks: List[np.ndarray],
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
      
        batch_size = y_true.shape[0]
        grads_W = [None] * len(self.weights)
        grads_b = [None] * len(self.biases)

        # Compute derivative of loss w.r.t output (dA) or dZ directly
        # If last activation is softmax with cross entropy, derivative simplifies
        last_act = self.activations[-1]
        loss_name = type(self.loss_fn).__name__
        if isinstance(self.loss_fn, CrossEntropyLoss) and self.activations[-1] == "softmax":
            # Not used; handle in else branch below
            pass

        # Compute gradient of output layer
        # If loss is cross entropy and activation is softmax, derivative is (y_pred - y_true)/batch_size
        # Otherwise, chain rule: dL/dZ = dL/dA * dA/dZ
        L = len(self.weights)
        if self.activations[-1] == "softmax" and isinstance(self.loss_fn, CrossEntropyLoss):
            dZ = activations[-1] - y_true  # (batch_size, n_classes), already averaged in derivative
            dZ /= batch_size
        else:
            dA = self.loss_fn.derivative(activations[-1], y_true)
            # Multiply by derivative of output activation
            act_name = self.activations[-1]
            Z = pre_activations[-1]
            if act_name == "linear":
                dZ = dA * linear_derivative(Z)
            elif act_name == "softmax":
                # For softmax with other loss functions (e.g. MSE), compute derivative manually
                # Compute Jacobian: but we approximate derivative as dA * derivative of softmax = dA * (softmax*(1-softmax))
                s = softmax(Z)
                dZ = dA * (s * (1 - s))
            else:
                dZ = dA * ACTIVATIONS[act_name][1](Z)
        
        # Loop over layers backwards
        for i in reversed(range(L)):
            A_prev = activations[i]  # activation from previous layer
            W = self.weights[i]

            # Compute gradients for W and b
            grads_W[i] = np.dot(A_prev.T, dZ)
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True)

            # Add L2 regularization gradient for weights
            if self.l2_lambda > 0:
                grads_W[i] += self.l2_lambda * W

            if i > 0:
                # Propagate gradient to previous layer: dA_prev = dZ @ W.T
                dA_prev = np.dot(dZ, W.T)
                # Apply dropout mask to gradient if used
                if self.use_dropout and dropout_masks[i - 1] is not None:
                    dA_prev *= dropout_masks[i - 1]
                # Multiply by derivative of activation
                Z_prev = pre_activations[i - 1]
                act_name = self.activations[i - 1]
                if act_name == "linear":
                    dZ = dA_prev * linear_derivative(Z_prev)
                elif act_name == "softmax":
                    s = softmax(Z_prev)
                    dZ = dA_prev * (s * (1 - s))
                else:
                    dZ = dA_prev * ACTIVATIONS[act_name][1](Z_prev)
        return grads_W, grads_b

    def update_parameters(self, grads_W: List[np.ndarray], grads_b: List[np.ndarray]) -> None:
        
        if self.optimizer == "sgd":
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_W[i]
                self.biases[i] -= self.learning_rate * grads_b[i]
        elif self.optimizer == "momentum":
            for i in range(len(self.weights)):
                self.velocities_W[i] = self.momentum * self.velocities_W[i] - self.learning_rate * grads_W[i]
                self.velocities_b[i] = self.momentum * self.velocities_b[i] - self.learning_rate * grads_b[i]
                self.weights[i] += self.velocities_W[i]
                self.biases[i] += self.velocities_b[i]
        elif self.optimizer == "adam":
            self.t += 1
            for i in range(len(self.weights)):
                # Update biased first moment estimate
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                # Update biased second raw moment estimate
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (grads_W[i] ** 2)
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)
                # Compute bias-corrected first and second moment estimates
                m_W_hat = self.m_W[i] / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_W_hat = self.v_W[i] / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b[i] / (1 - self.beta2 ** self.t)
                # Update parameters
                self.weights[i] -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
                self.biases[i] -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True,
        shuffle: bool = True,
    ) -> List[float]:
        
        N = X.shape[0]
        history = []
        for epoch in range(1, epochs + 1):
            if shuffle:
                idx = np.random.permutation(N)
                X = X[idx]
                y = y[idx]
            epoch_loss = 0.0
            for start in range(0, N, batch_size):
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                # Forward pass
                activations, pre_acts, dropout_masks = self.forward(X_batch, training=True)
                # Compute loss
                loss_val = self.loss_fn(activations[-1], y_batch)
                epoch_loss += loss_val * X_batch.shape[0]
                # Backward pass
                grads_W, grads_b = self.backward(activations, pre_acts, y_batch, dropout_masks)
                # Update parameters
                self.update_parameters(grads_W, grads_b)
            epoch_loss /= N
            history.append(epoch_loss)
            if verbose and (epoch % max(1, epochs // 20) == 0 or epoch == 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.6f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities or regression outputs for given input."""
        activations, _, _ = self.forward(X, training=False)
        return activations[-1]

    def accuracy(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute accuracy for classification tasks."""
        pred_labels = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        return np.mean(pred_labels == true_labels)


# -----------------------------------------------------------------------------
# Synthetic dataset generation for demonstration
# -----------------------------------------------------------------------------
def generate_synthetic_circle_dataset(num_samples: int = 1000, noise: float = 0.1, seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
  
    if seed is not None:
        np.random.seed(seed)
    # Generate random points in a square [-1, 1] x [-1, 1]
    X = np.random.uniform(-1.0, 1.0, (num_samples, 2))
    # Compute distance from origin
    r = np.sqrt(np.sum(X ** 2, axis=1))
    # Label: 1 if inside circle of radius 0.5, else 0
    labels = (r < 0.5).astype(int)
    # Add noise to points
    X += np.random.normal(0, noise, X.shape)
    # One-hot encode labels
    y = np.zeros((num_samples, 2))
    y[np.arange(num_samples), labels] = 1
    return X, y


def parse_args(args: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an advanced neural network from scratch using NumPy.")
    parser.add_argument('--layer-sizes', type=int, nargs='+', required=True,
                        help='List of layer sizes, including input and output dimensions. e.g. --layer-sizes 2 16 8 2')
    parser.add_argument('--activations', type=str, nargs='+', required=True,
                        help='Activation functions for each layer except input. e.g. --activations relu relu softmax')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=list(LOSSES.keys()),
                        help='Loss function to use (mse or cross_entropy)')
    parser.add_argument('--init', type=str, default='he', choices=['he', 'xavier', 'normal'],
                        help='Weight initialization method')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'momentum', 'adam'],
                        help='Optimizer type')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate for optimizer')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum factor (for momentum optimizer)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 parameter for Adam')
    parser.add_argument('--beta2', type=float, default=0.999, help='Beta2 parameter for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Epsilon parameter for Adam')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization strength')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability for hidden layers')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Mini-batch size')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--demo', action='store_true', help='Use synthetic dataset for demonstration')
    return parser.parse_args(args)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Generate synthetic dataset for demonstration if requested
    if args.demo:
        X, y = generate_synthetic_circle_dataset(num_samples=2000, noise=0.1, seed=args.seed)
        input_dim = X.shape[1]
        output_dim = y.shape[1]
        # Validate provided layer sizes against dataset
        if args.layer_sizes[0] != input_dim or args.layer_sizes[-1] != output_dim:
            print(f"[Warning] Adjusting layer sizes from {args.layer_sizes} to match dataset dimensions {input_dim}->{output_dim}")
            layer_sizes = [input_dim] + args.layer_sizes[1:-1] + [output_dim]
        else:
            layer_sizes = args.layer_sizes
    else:
        print("Please provide your own dataset if not using --demo flag.")
        sys.exit(1)

    # Validate activations
    if len(args.activations) != len(layer_sizes) - 1:
        raise ValueError(f"Number of activation functions ({len(args.activations)}) must be {len(layer_sizes) - 1} to match layers")

    # Instantiate the neural network
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        activations=args.activations,
        loss=args.loss,
        weight_init=args.init,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        momentum=args.momentum,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        l2_lambda=args.l2,
        dropout_prob=args.dropout,
    )

    # Train the network
    history = nn.fit(X, y, epochs=args.epochs, batch_size=args.batch_size, verbose=True)

    # Evaluate on training data
    y_pred = nn.predict(X)
    if args.loss == 'cross_entropy':
        acc = nn.accuracy(y_pred, y)
        print(f"Training accuracy: {acc * 100:.2f}%")
    else:
        mse = np.mean((y_pred - y) ** 2)
        print(f"Training MSE: {mse:.6f}")

    # Optionally print final loss history or save model here


if __name__ == '__main__':
    main(sys.argv[1:])