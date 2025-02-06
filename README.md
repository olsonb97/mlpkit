# nnkit

This is a simple machine learning kit designed to make tinkering with neural network concepts easy and approachable. Key goals of this project are to abstract the math, and modularize the neural network building process, while keeping the building process explicit enough to experiment with. This is pretty much **only** for tinkering and education, as it uses manual differentiation and is not designed for efficiency. Pytorch or Tinygrad will suit actual projects much better.

**Currently, only MLP classification tasks are supported. There is no support for unsupervised learning, CNNs, etc.**

## Features

- **Initialization**: Supports He, Xavier (normal and uniform), and customs can be added.
- **Activation**: Includes ReLU, Leaky ReLU, Tanh, Softmax, and Identity functions with respective derivatives.
- **CPU & GPU Support**: Specify device to run either NumPy (CPU) or CuPy (GPU).
- **Multilayer Perceptron**: Implements fully connected layers where each neuron connects to every neuron in the next layer.
- **Training**: Supports L2 regularization and backpropagation. Encodes one-hot labels during training.
- **Visualization**: Built-in plotting for loss and learning rate decay.

## Installation

nnkit requires Python and these dependencies:

- `numpy` (for matrix operations)
- `cupy` (for GPU acceleration)
- `matplotlib` (for visualization)

To install dependencies:

```sh
pip install numpy cupy matplotlib
```

## Usage

### Creating a Model

```python
from nn import MLP, ReluLayer, SoftmaxLayer
from kit import Init, Loss

# Initialize model
model = MLP("gpu")

# Add layers
model.add_layer(
    ReluLayer(
        input_size=784, 
        output_size=128,
        initialize=Init.he,
        l2_decay=0.1
    )
)
model.add_layer(
    SoftmaxLayer(
        input_size=128, 
        output_size=10,
        initialize=Init.xavier_normal
    )
)

# Train model
model.train(dataset, labels, epochs=20, loss_func=Loss.cross_entropy)
```

### Making Predictions

```python
predictions = model.predict(test_data)
accuracy = model.get_accuracy(predictions, test_labels)
print(f"Test Accuracy: {accuracy:.2f}%")
```

## Saving and Loading Models
**Note**: Saving only includes weights and biases, not the model architecture. To load a model, manually reconstruct its layers before loading the saved parameters.

```python
model.save("params.npz")
model.load("params.npz")
```

## License
Licensed under MIT License