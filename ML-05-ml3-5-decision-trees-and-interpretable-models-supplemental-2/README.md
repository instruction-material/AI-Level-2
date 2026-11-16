# Decision stump: Gini impurity and threshold choice

Complete gini and best_split in the starter. Growth and prediction are supplied so this exercise focuses on one interpretable split.

The eight training x values are 0 through 7; y alternates 0 and 1. The first four labels are A and the last four B. The root impurity is 1/2. Splitting feature x at 3.5 gives two pure four-row children, weighted impurity 0 and gain 1/2. Splitting y gives two mixed groups and no gain. Held-out x values 0.5,2.5,4.5,6.5 predict A,A,B,B at both depths, accuracy 1.0.

Compare the x=3.5 and y=0.5 candidates by hand. Trace queries exactly at x=3.5 and just above it. Duplicate x as the second feature and verify the declared feature-index tie rule. Explain why increasing depth cannot improve this already pure fitted tree.

Each role includes fixed train.csv and test.csv files, read relative to main.py. Use Python 3.10 or later and its standard library. The fixture describes fictional measurements and labels.

For a node with class proportions p, Gini impurity is 1-sum(p*p). Candidate splits use numeric midpoints between distinct sorted training values and route values <= threshold left. Minimize the sample-count-weighted child impurity. Gains within 1e-12 count as ties: choose the lowest feature index, then the lowest threshold. Leaves use the largest class count; tied classes choose the alphabetically first label.

Fit uses training data only. Nodes stop at max_depth, a pure class, no distinct split, or no impurity improvement greater than 1e-12. This greedy positive-gain rule can stop on XOR even though a deeper tree could represent XOR. Depth 0 means a single majority leaf; allowed depths are 0 through 8 for this small teaching implementation. Root depth is 0.

Run python3 main.py in a role folder or use its absolute path from any directory. Both depth-1 and depth-2 predictions are printed. Use trace(model, query) to inspect the exact numeric comparisons and final training class counts. Those counts describe the fitted leaf and do not establish real-world confidence or causal explanations.

Keep a hand calculation of the selected split, one traced prediction and the held-out predictions. Test rows remain outside fitting. A tiny perfect exercise score does not establish generalization; choose depth on a separate validation split when using a larger dataset.

Run python3 -I -B test_tree.py from the pack root for the reference checks.

[Algorithm reference](https://scikit-learn.org/stable/modules/tree.html).
