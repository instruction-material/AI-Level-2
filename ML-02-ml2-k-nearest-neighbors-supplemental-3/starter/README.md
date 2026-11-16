# Starter


Use the supplied fictional points 0, A; 2, A; 3, B on one feature axis. The y column is always 0 and contributes no distance. Four fixed test queries compare two vote rules on the same training data.

Implement predict in the starter using the supplied nearest function. Uniform voting gives each selected neighbor one vote. Distance voting gives weight proportional to 1/d. Multiplying every vote by the same positive value preserves the result, so closest_distance/d avoids a reciprocal overflow.

If selected neighbors include distance 0, only those exact matches vote, each with weight 1. A tied label score chooses the alphabetically first label. Equal neighbor distances at the k boundary use coordinate order, then label order. This is the pack's declared teaching tie rule.

Run main.py from the selected starter or solution folder. Its fixture path is relative to the script and works from another directory too.

## Worked check

For query 2.9 and k=3, uniform votes are A: 2 and B: 1, so A wins. Inverse-distance weights are approximately A: 1/2.9+1/0.9=1.455939 and B: 1/0.1=10, so B wins. An exact query at 3 predicts B with distance weights even though the other two labels are A.

For test queries 2.9, 0.1, 2.1, 3.5, uniform predictions are A, A, A, A and distance predictions are B, A, A, B. Against the supplied fictional expected labels, the accuracies are 0.5 and1.0. This tiny constructed comparison explains the voting rule and does not establish that weighting is generally better.

## Complete the exercise

1. Write both score tables for2.9 before running the code.
2. Implement the uniform and distance modes with the declared tie rules.
3. Verify the four fixed queries and report both prediction lists.
4. Check an exact match and multiple contradictory exact matches; farther points cannot override the exact-match vote.
5. Check an invalid weight name, invalid k and a nonfinite coordinate.
6. Explain the constructed fixture's limitation and keep its test rows outside model selection.

Keep the hand-computed score table, observed predictions and a short comparison of the two vote rules.

Algorithm reference: https://scikit-learn.org/stable/modules/neighbors.html
