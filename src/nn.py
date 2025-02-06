import numpy as np
import cupy as cp
from src.plot import Plot
from src.kit import Init, Activate

def _verify(*args, xp=None):
    """Verifies that args are of a certain array type"""
    for arg in args:
        if not isinstance(arg, xp.ndarray):
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
        logits = self.xp.matmul(self.input_cache, self.weights) + self.biases
        return self.activation(logits, xp=self.xp)

    def backward(self, dL_da, learning_rate):
        dL_da = _convert(dL_da, self.xp)
        # Chain rule to find nested derivatives
        logits = self.xp.matmul(self.input_cache, self.weights) + self.biases

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
        dL_dx = self.xp.matmul(dL_dz, self.weights.T)

        # L2 regularization
        dL_dw += self.l2_decay * self.weights

        # Update weights and biases
        self.weights -= learning_rate * dL_dw
        self.biases -= learning_rate * dL_db

        return dL_dx

class ReluLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.relu
        self.derivative = Activate.relu_derivative

class EluLayer(Layer):
    def __init__(self, input_size, output_size, initialize=Init.basic, l2_decay=0):
        super().__init__(input_size, output_size, initialize, l2_decay)
        self.activation = Activate.elu
        self.derivative = Activate.elu_derivative

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
        layer.biases = self.xp.zeros(
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

    def backward_pass(self, dL_da, learning_rate):
        """Adjust weights per layer with derivatives"""
        for layer in reversed(self.layers):
            dL_da = layer.backward(dL_da, learning_rate)

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
            _verify(dataset, labels, xp=self.xp) # Check whether they are arrays
            if not self.layers:
                raise RuntimeError("MLP has no layers!")
            if not dataset.ndim == 2:
                raise ValueError(f"Dataset must be of '2' dimensions! Got: {dataset.ndim}")
            if not labels.ndim == 1:
                raise ValueError(f"Labels must be of '1' dimension! Got: {dataset.ndim}")
            
            # Initialize some things
            dataset = _convert(dataset, self.xp)
            labels = _convert(labels, self.xp)
            num_samples = dataset.shape[0]
            one_hot = self.xp.eye(len(self.xp.unique(labels)))[labels]
            batch_size = num_samples // batches

            # Values for matplotlib (requires np)
            losses = np.empty(shape=(epochs,), dtype=np.float32)
            learning_rates = np.empty(shape=(epochs,), dtype=np.float32)
            weights = np.empty(shape=(epochs,), dtype=np.float32)
            biases = np.empty(shape=(epochs,), dtype=np.float32)

            # Iterate epochs
            for epoch in range(epochs):
                learning_rate = learning_rate / (1 + decay_rate * epoch)
                if shuffle:
                    indices = self.xp.random.permutation(num_samples)
                    dataset, one_hot = dataset[indices], one_hot[indices]

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
                    self.backward_pass(dL_da, learning_rate)

                # Track data
                losses[epoch] = epoch_loss
                learning_rates[epoch] = learning_rate
                weights[epoch] = np.sum(sum(layer.weights.mean() for layer in self.layers))
                biases[epoch] = np.sum(sum(layer.biases.mean() for layer in self.layers))

                if print_interval and epoch % print_interval == 0:
                    print(f"Epoch {epoch+1}/{epochs}: Loss = {epoch_loss:.8f}")

        except KeyboardInterrupt:
            pass # Allow canceling

        if plot:
            self.plot_training(
                epochs=epochs,
                losses=losses,
                learning_rates=learning_rates,
                weights=weights,
                biases=biases
            )
            self.show()

    def get_accuracy(self, predicted_labels, true_labels):
        """Returns an accuracy percentage between predicted and true labels"""
        _verify(predicted_labels, true_labels, xp=self.xp)
        predicted_labels = _convert(predicted_labels, self.xp)
        true_labels = _convert(true_labels, self.xp)
        return self.xp.mean(predicted_labels == true_labels) * 100

    def predict(self, dataset):
        """Returns a label prediction for given dataset"""
        _verify(dataset, xp=self.xp)
        probs = self.forward_pass(dataset)
        return self.xp.argmax(probs, axis=1)

    def test(self, dataset, labels):
        """Returns an accuracy percentage on given dataset and labels"""
        _verify(dataset, labels, xp=self.xp)
        predicted_labels = self.predict(dataset)
        true_labels = _convert(labels, self.xp)
        return self.get_accuracy(predicted_labels, true_labels)

    def save(self, filename="params.npz"):
        """Saves weights and biases to a custom npz structure"""
        params = {}
        for i, layer in enumerate(self.layers):
            # Convert all to numpy
            w = _convert(layer.weights, np)
            b = _convert(layer.biases, np)
            params[f"layer_{i}_weights"] = w
            params[f"layer_{i}_biases"] = b
        np.savez(filename, **params)

    def load(self, filename="params.npz"):
        """Loads weights and biases from a custom npz structure"""
        data = np.load(filename)
        for i, layer in enumerate(self.layers):
            weights = data[f"layer_{i}_weights"]
            biases = data[f"layer_{i}_biases"]
            if layer.xp is cp:
                layer.weights = cp.asarray(weights)
                layer.biases = cp.asarray(biases)
            else:
                layer.weights = weights
                layer.biases = biases