# Hidden-layer network: backpropagation and XOR

Complete forward and loss_and_gradient in the starter. The supplied seed 17 initialization breaks hidden-unit symmetry, and the full-batch loop updates all parameters.

The model has two inputs, three tanh hidden units and one sigmoid output. Parameters [0:3], [3:6] and [6:9] are each hidden unit's [w_x,w_y,b]. Parameters [9:12] are the three output weights; parameter 12 is the output bias. A hidden value is h_j=tanh(w_jx*x+w_jy*y+b_j). The output logit is sum(v_j*h_j)+b_out.

For one row, delta_out=p-y. Output-weight derivatives are delta_out*h_j. Each hidden preactivation derivative is delta_out*v_j*(1-h_j*h_j); multiply this by x or y for its two input-weight derivatives, and use it directly for the hidden bias.

At all-zero parameters and the balanced XOR corners, probabilities are 1/2, loss is log(2), and every gradient is 0. This symmetry trap explains why training starts from small seeded random weights. With the supplied seed 17 and 5,000 steps, the reference predicts 0,1,1,0 for the four held-out near-corner queries. Check gradients against centered finite differences before training. Compare the hidden network's boundary with the preceding single-neuron baseline and explain why other initializations or datasets can converge differently.

Use Python 3.10 or later and its standard library. Each role includes fixed train.csv and test.csv files; paths are relative to main.py. All features are bounded to [0,1] and labels are numeric 0 or 1. These are authored Boolean-pattern exercises.

A logit z becomes probability sigmoid(z). Use the stable two-branch sigmoid implementation and mean binary cross-entropy computed as max(z,0)-y*z+log1p(exp(-abs(z))). For a sigmoid output and this loss, the derivative with respect to z is p-y. Average every parameter derivative over the training rows before updating theta <- theta - learning_rate * gradient.

Training is deterministic full-batch gradient descent with learning_rate=0.5 by default. The exercise accepts 1 through 10,000 steps and a rate in (0,1]. Fit only the four training corners. Keep the four near-corner test queries outside training. Prediction uses probability >= 0.5 as class 1, otherwise 0. A small fixture's accuracy checks model behavior; it does not establish performance on larger data.

Run python3 main.py from either role folder or use its absolute path from another directory. Keep the forward/gradient hand calculation, initial and final training loss, and held-out predictions. Explain the distinct role of a learning rate, training iterations and the classification threshold.


[Algorithm reference](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

Role: starter.
