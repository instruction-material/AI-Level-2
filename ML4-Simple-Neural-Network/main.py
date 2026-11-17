"""Worked forward pass: two hidden sigmoid neurons and one ReLU output."""
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


def relu(x):
    return max(0, x)


def simulate(x1=0, x2=1, b=1):
    """Return the two hidden activations and final output without fitting weights."""
    hidden_neuron1 = Neuron(1, 2, sigmoid)
    hidden_neuron2 = Neuron(3, 1, sigmoid)
    output_neuron = Neuron(3, 5, relu)
    hidden1_output = hidden_neuron1.run(x1, x2, b)
    hidden2_output = hidden_neuron2.run(x1, x2, b)
    output = output_neuron.run(hidden1_output, hidden2_output, b)
    return hidden1_output, hidden2_output, output


if __name__ == "__main__":
    for value in simulate():
        print(value)
