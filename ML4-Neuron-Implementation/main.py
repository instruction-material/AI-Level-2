"""Worked two-input neuron. Python 3.10+; standard library only."""
import math


class Neuron:
    """A two-input weighted sum followed by the supplied activation."""

    def __init__(self, w1, w2, activ_fn):
        self.w1 = w1
        self.w2 = w2
        self.activation_function = activ_fn

    def run(self, x1, x2, b):
        return self.activation_function(x1 * self.w1 + x2 * self.w2 + b)


def sigmoid(x):
    """Return the logistic activation without exponentiating a large positive value."""
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1 + exp_x)


if __name__ == "__main__":
    n = Neuron(1, 2, sigmoid)
    print(n.run(0, 1, 1))
