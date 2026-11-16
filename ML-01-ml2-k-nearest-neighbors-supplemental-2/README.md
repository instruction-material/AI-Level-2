# KNN: training-only scaling and uniform voting

This supplied parcel-sorting fixture has two measured features, x and y, and fictional classes A and B. The eight training rows fit the standardizer and classifier. Four predefined test rows stay outside every fit operation.

Implement fit_scale, transform and predict in the starter. The provided nearest function validates the input and selects neighbors. Fit the mean and population standard deviation on training rows only; a constant column uses scale 1. Transform training and test rows with those same fitted values.

Each selected neighbor casts one vote. A tied vote chooses the alphabetically first label. Equal distances at the k boundary use coordinate order, then label order. This declared teaching tie rule can differ from a library's training-row order.

Run main.py from the selected starter or solution folder. Paths are relative to the script, so the same command also works from another directory with an absolute script path.

## Worked check

Both training means are 5. Each training feature has population variance 17, so its scale is sqrt(17), approximately 4.123106. For test point (5,5), the two closest rows are (2,2),A and (8,8),B. The next equal-distance rows are selected by coordinate order; k=3 votes A. The four predictions are A, B, A, B for both k=1 and k=3. The tiny supplied fixture's k=3 accuracy is 1.0; this is a checkable exercise result, not evidence of general performance.

## Complete the exercise

1. Compute the two means and scales before opening the four test labels.
2. Trace the selected distances and votes for(1,1)and (5,5).
3. Run k=1 and k=3 on the same four test rows and compare their predictions with the fixed expected labels.
4. Transform a separate far-away query without changing the fitted means or scales.
5. Check a constant column, a tied vote and invalid k. Reordering equal-distance training rows must preserve the declared coordinate tie rule.

Keep the distance/vote trace, prediction list and the reason held-out rows cannot influence scaling. The optional next pack compares uniform and inverse-distance voting.

Algorithm reference: https://scikit-learn.org/stable/modules/neighbors.html
