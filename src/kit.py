import numpy as np
import cupy as cp

def validate_device(func):
    def wrapper(*args, **kwargs):
        device = kwargs.pop('device', None)
        if device not in (np, cp):
            raise ValueError(f"Invalid device: '{device}'. Please Numpy as 'np' or Cupy as 'cp'")
        return func(*args, device=device, **kwargs)
    return wrapper

class Init:
    @validate_device
    def basic(input_size, output_size, range=(-0.1, 0.1), device=None):
        return device.random.uniform(range[0], range[1], size=(input_size, output_size))

    @validate_device
    def he(input_size, output_size, device=None):
        std = device.sqrt(2 / input_size)
        return device.random.normal(0, std, size=(input_size, output_size))

    @validate_device
    def xavier_normal(input_size, output_size, device=None):
        std = device.sqrt(1 / (input_size + output_size))
        return device.random.normal(0, std, size=(input_size, output_size))

    @validate_device
    def xavier_uniform(input_size, output_size, device=None):
        limit = device.sqrt(6 / (input_size + output_size))
        return device.random.uniform(-limit, limit, size=(input_size, output_size))

class Activate:
    @validate_device
    def relu(x, device=None):
        return device.maximum(0, x)

    @validate_device
    def relu_derivative(x, device=None):
        return device.where(x > 0, 1, 0)

    @validate_device
    def leaky_relu(x, alpha=0.01, device=None):
        return device.where(x > 0, x, alpha * x)

    @validate_device
    def leaky_relu_derivative(x, alpha=0.01, device=None):
        return device.where(x > 0, 1, alpha)

    @validate_device
    def tanh(x, device=None):
        return device.tanh(x)

    @validate_device
    def tanh_derivative(x, device=None):
        return 1 - device.tanh(x) ** 2

    @validate_device
    def identity(x, device=None):
        return x

    @validate_device
    def identity_derivative(x, device=None):
        return device.ones_like(x)

    @validate_device
    def softmax(z, device=None):
        # Handle vectors and matrices
        if z.ndim == 1:
            z = z.reshape(1, -1)
            vector = True
        else:
            vector = False

        stable_logits = z - device.max(z, axis=1, keepdims=True)
        numerator = device.exp(stable_logits)
        denominator = device.sum(numerator, axis=1, keepdims=True)
        result = numerator / denominator
        return result.flatten() if vector else result

    @validate_device
    def softmax_derivative(p, device=None):
        # Jacobian matrix. Not recommended to use, especially in output layer.
        # More efficient to just do dL/da = dL/dz since softmax scales
        p = p.reshape(-1, 1)
        return device.diagflat(p) - device.dot(p, p.T)

class Loss:
    @validate_device
    def cross_entropy(p, y, gradient=False, device=None):
        # Relies on p being probabilities. Doesn't work with logits
        epsilon = 1e-10
        loss = -device.mean(device.sum(y * device.log(device.clip(p, epsilon, 1.0)), axis=1))

        if gradient:
            return loss, (p - y) / p.shape[0]

        return loss
