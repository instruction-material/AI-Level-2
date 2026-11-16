# Bounded decision tree: recursion, depth and prediction paths

Complete grow in the starter. Gini calculation and split selection are supplied from the preceding exercise. Recursively construct both children, keeping each node's training counts and obeying every stopping condition.

The eight training x values are 0 through 7; y alternates 0 and 1. Labels are A,A,B,B,B,B,A,A. Root impurity is 1/2. Splits x<=1.5 and x<=5.5 tie at weighted impurity 1/3; choose 1.5. The right child has four B and two A rows, impurity 4/9. At depth 2 its split x<=5.5 creates pure children.

For held-out x values 0.5,2.5,4.5,6.5, a depth-1 stump predicts A,B,B,B, accuracy 0.75. A depth-2 tree predicts A,B,B,A, accuracy 1.0. Trace x=6.5 through right,right to its A leaf. Contrast that path with the stump's mixed B leaf.

Compare depths on the fixed split, draw every resulting node and its class counts, then explain why choosing depth after seeing these four test labels would leak test information. Show how identical measurements, a pure node and XOR each stop under the declared positive-gain rule.

Each role includes fixed train.csv and test.csv files, read relative to main.py. Use Python 3.10 or later and its standard library. The fixture describes fictional measurements and labels.

For a node with class proportions p, Gini impurity is 1-sum(p*p). Candidate splits use numeric midpoints between distinct sorted training values and route values <= threshold left. Minimize the sample-count-weighted child impurity. Gains within 1e-12 count as ties: choose the lowest feature index, then the lowest threshold. Leaves use the largest class count; tied classes choose the alphabetically first label.

Fit uses training data only. Nodes stop at max_depth, a pure class, no distinct split, or no impurity improvement greater than 1e-12. This greedy positive-gain rule can stop on XOR even though a deeper tree could represent XOR. Depth 0 means a single majority leaf; allowed depths are 0 through 8 for this small teaching implementation. Root depth is 0.

Run python3 main.py in a role folder or use its absolute path from any directory. Both depth-1 and depth-2 predictions are printed. Use trace(model, query) to inspect the exact numeric comparisons and final training class counts. Those counts describe the fitted leaf and do not establish real-world confidence or causal explanations.

Keep a hand calculation of the selected split, one traced prediction and the held-out predictions. Test rows remain outside fitting. A tiny perfect exercise score does not establish generalization; choose depth on a separate validation split when using a larger dataset.

Run python3 -I -B test_tree.py from the pack root for the reference checks.

[Algorithm reference](https://scikit-learn.org/stable/modules/tree.html).
