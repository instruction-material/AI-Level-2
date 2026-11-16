# Multinomial Naive Bayes: training-only vocabulary and log scores

The six fictional training messages and four held-out messages are supplied in each role's data folder. They contain no personal correspondence. Use Python 3.10 or later and its standard library.

Complete fit and log_scores in starter/main.py. Tokenization lowercases text and selects ASCII alphabetic words. Fit the vocabulary and class token counts on training messages only. Class priors count documents. Within each class, the denominator is its training token total + alpha * vocabulary size; each token numerator is its count + alpha. Alpha defaults to 1 and must be positive.

Sum log(prior) and query token counts times log(likelihood). Repeated words contribute repeatedly. Ignore held-out words absent from the fitted vocabulary; they never change the model. Empty or entirely unknown messages use the class priors. An exact score tie chooses the alphabetically first label.

## Worked check

The vocabulary has six words. Both classes have three documents and six tokens; every class-specific word occurs twice in its class and zero times in the other. With alpha=1, each matching likelihood is 3/12 and each nonmatching likelihood is 1/12.

For "claim prize", spam has unnormalized mass (1/2)*(3/12)^2=1/32 and ham has (1/2)*(1/12)^2=1/288. Normalizing gives spam 0.9 and ham 0.1. These are model probabilities under its assumptions, not a guarantee of real-world calibration. Four held-out predictions are ham,spam,ham,spam, accuracy 1.0 on this toy fixture.

1. Trace token counts and log scores for "claim prize" by hand.
2. Show that "unseen" never enters the trained vocabulary.
3. Predict a message with "prize" repeated 2000 times. Direct probability products underflow; log scores stay finite.
4. Compare alpha=1 with alpha=2 on the same held-out split.
5. Describe how correlated words and repeated wording can exaggerate evidence. This exercise is a local model calculation, not a deployed spam filter.

Run python3 main.py from a role folder, or use its absolute path from any directory. Each role is self-contained. Keep the fixed vocabulary, hand calculation, predictions and explanation of the independence assumption.

Run python3 -I -B test_naive_bayes.py from this pack's root for the reference checks.

Role: solution.

[Algorithm reference](https://scikit-learn.org/stable/modules/naive_bayes.html).
