"""Build a two-input neuron. Python 3.10+; standard library only."""
import math


class Neuron:
    def __init__(self, w1, w2, activ_fn):
        """Store both weights and the activation function on this instance."""
        raise NotImplementedError("Store the Neuron weights and activation function.")

    def run(self, x1, x2, b):
        """Return activ_fn(x1*w1 + x2*w2 + b) using this neuron's stored values."""
        raise NotImplementedError("Implement the Neuron weighted sum and activation.")


def sigmoid(x):
    """Return the logistic activation without exponentiating a large positive value."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


if __name__ == "__main__":
    n = Neuron(1, 2, sigmoid)
    print(n.run(0, 1, 1))
