# Two-input neuron

This folder is the learner starter. Use Python 3.10 or later with its standard library.

Complete `Neuron.__init__` and `Neuron.run` in the starter. Store `w1`, `w2` and activ_fn, then compute `z=x1*w1+x2*w2+b` and return the activation of z. The supplied stable sigmoid handles large positive and negative logits.

Run `python3 main.py` from this folder, or run its absolute path from another directory. The untouched starter raises `NotImplementedError` at its named task. The reference prints approximately 0.9525741268224334.

For `x1=0, x2=1`, weights 1 and 2, and bias 1, `z=3` and `sigmoid(z)=1/(1+exp(-3))`. Change `w2` to 1: `z=2` and the output is approximately 0.8807970779778823. Changing only `w1` has no effect while `x1=0`. Explain the effect of changing bias by one.

Supply a different callable activation, such as max(0,z), through the constructor. Check `z=-2` with ReLU: the output is 0. Inspect the reference after attempting the class. This exercise computes a forward value; it does not fit weights.
