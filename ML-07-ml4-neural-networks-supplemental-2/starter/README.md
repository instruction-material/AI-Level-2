# Single sigmoid neuron: forward pass and loss gradient

Complete loss_and_gradient in the starter. The forward pass and training loop are supplied. Parameters are [w_x,w_y,b], giving logit w_x*x+w_y*y+b.

The training labels follow OR: (0,0)->0 and the other corners->1. At the all-zero initialization, each probability is 1/2 and mean loss is log(2), about 0.693147. The average gradient is [-1/4,-1/4,-1/4], so one update at rate 1/2 gives [1/8,1/8,1/8].

Trace that first update, compare one step with 2,000 steps, and evaluate the four held-out near-corner queries. The default reference predicts 0,1,1,1. This neuron is a logistic-regression baseline with a linear logit boundary; explain why it cannot separate XOR.

Use Python 3.10 or later and its standard library. Each role includes fixed train.csv and test.csv files; paths are relative to main.py. All features are bounded to [0,1] and labels are numeric 0 or 1. These are authored Boolean-pattern exercises.

A logit z becomes probability sigmoid(z). Use the stable two-branch sigmoid implementation and mean binary cross-entropy computed as max(z,0)-y*z+log1p(exp(-abs(z))). For a sigmoid output and this loss, the derivative with respect to z is p-y. Average every parameter derivative over the training rows before updating theta <- theta - learning_rate * gradient.

Training is deterministic full-batch gradient descent with learning_rate=0.5 by default. The exercise accepts 1 through 10,000 steps and a rate in (0,1]. Fit only the four training corners. Keep the four near-corner test queries outside training. Prediction uses probability >= 0.5 as class 1, otherwise 0. A small fixture's accuracy checks model behavior; it does not establish performance on larger data.

Run python3 main.py from either role folder or use its absolute path from another directory. Keep the forward/gradient hand calculation, initial and final training loss, and held-out predictions. Explain the distinct role of a learning rate, training iterations and the classification threshold.


[Algorithm reference](https://scikit-learn.org/stable/modules/neural_networks_supervised.html).

Role: starter.
