import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Plot:
    def __init__(self):
        pass

    def _plot_loss_journey(self, weights, biases, losses, epochs):
        weights = list(weights)
        biases = list(biases)
        losses = list(losses)
        fig = plt.figure(figsize=(10,8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')

        x_data, y_data, z_data = [], [], []
        line, = ax.plot([], [], [], 'r-', lw=2)

        ax.set_xlim(min(weights), max(weights))
        ax.set_ylim(min(biases), max(biases))
        ax.set_zlim(0, 2.4)

        def update(frame):
            x, y, z = weights[frame], biases[frame], losses[frame]
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)

            # Update line
            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)

            return line,

        FuncAnimation(fig, update, frames=epochs, interval=100, blit=False, repeat=False)
        plt.show()

    def plot_training(self, losses, epochs, learning_rates, weights, biases):
        self._plot_loss_journey(weights, biases, losses, epochs)