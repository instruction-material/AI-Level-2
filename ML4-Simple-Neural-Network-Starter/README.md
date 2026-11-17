# Tiny neural-network forward pass

This folder is the learner starter. Use Python 3.10 or later with its standard library. Neuron, stable sigmoid and ReLU are supplied.

Complete `simulate` in the starter. Create hidden sigmoid neurons with weights (1,2) and (3,1), and a ReLU output neuron with weights (3,5). Use `x1=0, x2=1` and bias `b=1` at every neuron. Return a tuple `(hidden1_output, hidden2_output, output)`.

The first hidden logit is `0*1+1*2+1=3`; `sigmoid(3)` is approximately 0.9525741268224334. The second is `0*3+1*1+1=2`; `sigmoid(2)` is approximately 0.8807970779778823. The output logit is `3*hidden1+5*hidden2+1`, approximately 8.261707770356711; ReLU preserves this positive value. These are fixed forward-pass computations, not training results or class probabilities.

Run `python3 main.py` from this folder, or run its absolute path from any directory. The untouched starter raises `NotImplementedError` at `simulate`. The reference prints the three values on separate lines. Change x2 from 1 to 0, trace both hidden logits and the final output, then compare the printed intermediate values with the hand calculation.

Try a sigmoid output activation while preserving the hidden values. Explain its (0,1) range compared with ReLU's nonnegative unbounded range. A classification threshold would be an additional declared rule. Inspect the reference after attempting the wiring.
