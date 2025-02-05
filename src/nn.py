import numpy as np
import cupy as cp
from src.plot import Plot
from src.kit import Init, Activate

def _verify(*args):
    """Verifies that args are of np or cp"""
    for arg in args:
        if not isinstance(arg, (np.ndarray, cp.ndarray)):
            raise ValueError(f"Expected datatype was an array! Instead got: {type(arg)}")

def _convert(x, xp):
    """Converts array x to given np or cp"""
    if isinstance(x, (np.ndarray, cp.ndarray)):
        if xp is cp and isinstance(x, np.ndarray):
            return cp.asarray(x)
        elif xp is np and isinstance(x, cp.ndarray):
            return x.get()
        else:
            return x
    return xp.array(x) # If neither

class Layer:
    def __init__(self, input_size, output_size, initialize, l2_decay):
        self.input_size = input_size
        self.output_size = output_size
        self.initialize = initialize
        self.l2_decay = l2_decay

    def forward(self, x):
        x = _convert(x, self.xp)
        self.input_cache = self.xp.array(x)
        logits = self.xp.matmul(self.input_cache, self.weights) + self.bias
        return self.activation(logits, xp=self.xp)

    def backward(self, dL_da, learning_rate):
        dL_da = _convert(dL_da, self.xp)
        # Chain rule to find nested derivatives
        logits = self.xp.matmul(self.input_cache, self.weights) + self.bias

        # Get necessary derivatives
        if self.derivative is not None:
            da_dz = self.derivative(logits, xp=self.xp)
            dL_dz = dL_da * da_dz
        else: # Used when da_dz cancels out
            dL_dz = dL_da

        # Weight & Bias derivatives
        dL_dw = self.xp.matmul(self.input_cache.T, dL_dz)
        dL_db = self.xp.sum(dL_dz, axis=0)

        # Gradient to propagate backward
        self.dL_dx = self.xp.matmul(dL_dz, self.weights.T)

        # L2 regularization
        dL_dw += self.l2_decay * self.weights

        # Update weights and biases
        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db

        return self.dL_dx

class ReluLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.relu
        self.derivative = Activate.relu_derivative

class LeakyReluLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.leaky_relu
        self.derivative = Activate.leaky_relu_derivative

class TanhLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.tanh
        self.derivative = Activate.tanh_derivative

class LogitLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.identity
        self.derivative = Activate.identity_derivative

class SoftmaxLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.softmax
        self.derivative = None

class MLP(Plot):
    """Deep Learning Classification MLP"""
    def __init__(self, xp="cpu"):
        if xp not in ("cpu", "gpu"):
            raise ValueError(f"Invalid xp: '{xp}'. Use 'cpu' or 'gpu'")
        self.xp = np if xp == "cpu" else cp
        super().__init__()
        self.layers = []

    def add_layer(self, layer):
        """Add's a layer to the model's layer stack"""
        if not isinstance(layer, Layer):
            raise TypeError("MLP expected a Layer type object!")
        if self.layers and layer.input_size != self.layers[-1].output_size:
            raise ValueError(f"Layer with input_size '{layer.input_size}' does not match previous layer's output size!")
        layer.xp = self.xp
        layer.weights = layer.initialize(
            input_size=layer.input_size,
            output_size=layer.output_size,
            xp=self.xp
        ).astype(self.xp.float32)
        layer.bias = self.xp.zeros(
            shape=(layer.output_size,)
        ).astype(self.xp.float32)
        self.layers.append(layer)

    def del_layer(self, index=-1):
        """Removes a layer of the given index"""
        if not self.layers:
            raise ValueError("MLP has no layers to delete!")
        self.layers.pop(index)

    def forward_pass(self, x):
        """Predict input per layer"""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward_pass(self, dL_da, eta):
        """Adjust weights per layer with derivatives"""
        for layer in reversed(self.layers):
            dL_da = layer.backward(dL_da, eta)

    def train(
        self,
        dataset,
        labels,
        epochs,
        loss_func,
        batches=1,
        learning_rate=0.01,
        decay_rate=0.001,
        shuffle=True,
        plot=True,
        print_interval=None
    ):
        """Adjusts weights and biases of the model's layers by training on labeled data"""
        try:
            # Verify things
            _verify(dataset, labels) # Check whether they are arrays
            if not self.layers:
                raise RuntimeError("MLP has no layers!")
            if self.layers[-1].xp != self.xp:
                raise ValueError(
                    f"Found mismatch in MLP xp and output layer's xp!" +
                    "\nThese two must match for proper loss calculations.")
            if not dataset.ndim == 2:
                raise ValueError(f"Dataset must be of '2' dimensions! Got: {dataset.ndim}")
            if not labels.ndim == 1:
                raise ValueError(f"Labels must be of '1' dimension! Got: {dataset.ndim}")
            # Initialize some things
            dataset = _convert(dataset, self.xp)
            labels = _convert(labels, self.xp)
            num_samples = dataset.shape[0]
            one_hot = self.xp.eye(len(self.xp.unique(labels)))[labels]
            # np to make matplotlib easier
            losses = np.zeros(shape=(epochs,))
            learning_rates = np.zeros(shape=(epochs,))

            # Iterate epochs
            for epoch in range(epochs):
                eta = learning_rate / (1 + decay_rate * epoch)
                if shuffle:
                    indices = self.xp.random.permutation(num_samples)
                    dataset, one_hot = dataset[indices], one_hot[indices]

                batch_size = num_samples // batches
                epoch_loss = 0 # Aggregate batch loss

                # Iterate batches
                for i in range(batches):
                    # Get the proper batch
                    start = i * batch_size
                    end = (i + 1) * batch_size
                    batch_x = dataset[start:end]
                    batch_y = one_hot[start:end]

                    # Prediction
                    pred_y = self.forward_pass(batch_x)

                    # Gradient descent
                    loss, dL_da = loss_func(pred_y, batch_y, gradient=True, xp=self.xp)
                    epoch_loss += loss / batches # Proportion the loss

                    # Backpropagate
                    self.backward_pass(dL_da, eta)

                # Track data
                losses[epoch] = epoch_loss
                learning_rates[epoch] = eta

                if print_interval and epoch % print_interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.8f}")

        except KeyboardInterrupt:
            pass # Allow canceling

        if plot:
            self.plot_training(
                losses=losses,
                epochs=epochs,
                learning_rates=learning_rates
            )
            self.show()

    def get_accuracy(self, predicted_labels, true_labels):
        """Returns an accuracy percentage between predicted and true labels"""
        _verify(predicted_labels, true_labels)
        predicted_labels = _convert(predicted_labels, self.xp)
        true_labels = _convert(true_labels, self.xp)
        return self.xp.mean(predicted_labels == true_labels) * 100

    def predict(self, dataset):
        """Returns a label prediction for given dataset"""
        _verify(dataset)
        probs = self.forward_pass(dataset)
        return self.xp.argmax(probs, axis=1)

    def test(self, dataset, labels):
        """Returns an accuracy percentage on given dataset and labels"""
        _verify(dataset, labels)
        predicted_labels = self.predict(dataset)
        true_labels = _convert(labels, self.xp)
        return self.get_accuracy(predicted_labels, true_labels)

    def save(self, filename="params.npz"):
        """Saves weights and biases to a custom npz structure"""
        params = {}
        for i, layer in enumerate(self.layers):
            # Convert all to numpy
            w = _convert(layer.weights, np)
            b = _convert(layer.bias, np)
            params[f"layer_{i}_weights"] = w
            params[f"layer_{i}_bias"] = b
        np.savez(filename, **params)

    def load(self, filename="params.npz"):
        """Loads weights and biases from a custom npz structure"""
        data = np.load(filename)
        for i, layer in enumerate(self.layers):
            weights = data[f"layer_{i}_weights"]
            biases = data[f"layer_{i}_bias"]
            if layer.xp is cp:
                layer.weights = cp.asarray(weights)
                layer.bias = cp.asarray(biases)
            else:
                layer.weights = weights
                layer.bias = biases