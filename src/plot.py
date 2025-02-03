import matplotlib.pyplot as plt

class Plot:
    def __init__(self):
        self.fig = plt.figure(figsize=(10, 6))
        self.axes = [
            self.fig.add_subplot(2, 2, i+1) for i in range(2)
        ]

    def _plot_loss(self, epochs, losses):
        self.axes[0].plot(epochs, losses)
        self.axes[0].set_title("Loss Minimization")
        self.axes[0].set_xlabel("Epochs")
        self.axes[0].set_ylabel("Loss")

    def _plot_learning_rate(self, epochs, learning_rates):
        self.axes[1].plot(epochs, learning_rates)
        self.axes[1].set_title("Learning Rate Decay")
        self.axes[1].set_xlabel("Epochs")
        self.axes[1].set_ylabel("Learn Rate")

    def plot_training(self, losses, epochs, learning_rates):
        epochs = range(1, epochs+1)
        self._plot_loss(epochs, losses)
        self._plot_learning_rate(epochs, learning_rates)

    def show(self):
        plt.tight_layout()
        plt.show()
        plt.close(self.fig)