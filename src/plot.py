import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation

class Plot:
    def __init__(self):
        self.fig = plt.figure(figsize=(7, 7))
        self.gs = GridSpec(3, 2, figure=self.fig)
        self.axes = [
            self.fig.add_subplot(self.gs[0, 0]),
            self.fig.add_subplot(self.gs[0, 1]),
            self.fig.add_subplot(self.gs[1:, :], projection='3d')
        ]
        self.fig.subplots_adjust(left=0.05, right=0.5, top=0.5, bottom=0.05)


    def _plot_loss(self, epochs, losses):
        subplot = self.axes[0]
        subplot.plot(epochs, losses)
        subplot.set_title("Loss Minimization")
        subplot.set_xlabel("Epochs")
        subplot.set_ylabel("Loss")

    def _plot_learning_rate(self, epochs, learning_rates):
        subplot = self.axes[1]
        subplot.plot(epochs, learning_rates)
        subplot.set_title("Learning Rate Decay")
        subplot.set_xlabel("Epochs")
        subplot.set_ylabel("Learn Rate")

    def _plot_loss_journey(self, weights, biases, losses):
        subplot = self.axes[2]
        subplot.set_title("Loss Minimization")
        subplot.set_xlabel("Weights")
        subplot.set_ylabel("Biases")
        subplot.set_zlabel("Loss")
        weights = list(weights)
        biases = list(biases)
        losses = list(losses)

        x_data, y_data, z_data = [], [], []
        line, = subplot.plot([], [], [], 'r-', lw=2)
        scatter = subplot.scatter([], [], [], color='k', s=1)

        subplot.set_xlim(min(weights), max(weights))
        subplot.set_ylim(min(biases), max(biases))
        subplot.set_zlim(0, max(losses))
        subplot.text(weights[0], biases[0], losses[0], "Start", color='green')
        subplot.text(weights[-1], biases[-1], losses[-1], "End", color='blue')

        def update(frame):
            x, y, z = weights[frame], biases[frame], losses[frame]
            x_data.append(x)
            y_data.append(y)
            z_data.append(z)

            line.set_data(x_data, y_data)
            line.set_3d_properties(z_data)
            scatter._offsets3d = (x_data, y_data, z_data)
            return line, scatter

        return FuncAnimation(self.fig, update, frames=range(len(losses)), interval=100, blit=False, repeat=False)

    def plot_training(self, epochs, losses, learning_rates, weights, biases):
        epochs = range(1, epochs+1)
        self._plot_loss(epochs, losses)
        self._plot_learning_rate(epochs, learning_rates)
        self.anim = self._plot_loss_journey(weights, biases, losses)

    def show(self):
        plt.tight_layout()
        plt.show()
        plt.close(self.fig)