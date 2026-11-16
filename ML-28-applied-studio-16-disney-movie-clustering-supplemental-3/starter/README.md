# Assign a new movie with a frozen fit

Input is the retained course movie table from ML1-Disney-Movie-Clustering-Updated/disney.csv. Its source checksum is recorded at runtime. Original movie names and observed grosses are preserved; these are descriptive records, not causal or forecasting evidence. The retained source does not establish a license, so that provenance remains explicit.

## Complete the starter

Implement assign in model.py. Assign every row to its nearest supplied center by squared Euclidean distance. Resolve a tie using the lowest center index. A new query must reuse fitted means, scales, and centers without refitting. The remaining support functions and main.py are supplied. Read their declared input/output contracts before editing.

The bounded k-means pipeline standardizes the numeric features, uses squared-distance k-means++ initialization, compares seeds 7, 19 and 31, and caps each fit at 40 iterations. The final labels are recomputed against the returned centers even when the iteration cap is reached. An empty group is reseeded rather than divided by zero. Each result records the actual stop reason.

This case's experiment is frozen. Use the same row IDs when comparing partitions. Cluster numbers are arbitrary identifiers. Inertia is a sum of squared distances in the fitted feature space and can only be compared on the same rows and feature definitions. Silhouette uses ordinary Euclidean distances; for more than 120 rows its fixed, evenly spread index sample is approximate. An unsupervised discovery fit has no target labels: report geometry, resampling/seed stability and limitations rather than classification accuracy.

## Worked checks and verification

For rows (1,2) and (3,4), the centroid is (2,3). For columns [1,3], the population mean is 2 and scale is 1. Squared distances from (2,3) to (0,0) and (4,6) both equal 13, so the first center wins a tie. The partitions [0,0,1,1] and [9,9,4,4] have adjusted Rand 1 even though their label numbers differ. A single cluster has no defined silhouette.

1. Trace the function named above on a tiny hand-worked example.
2. Run the complete fixture and record row IDs, fitted feature definition, scaler, selected seed, inertia, silhouette and cluster profiles.
3. Explain the experiment's observed change and a limitation. Stable labels alone do not validate a real-world interpretation.
4. Restart from the same files and seeds, and compare the complete result. Test an empty input, a constant column, tied distances and an iteration-capped fit.
5. Use the returned centroids and scaler for a new query; keep that query outside the fit.

Use Python 3.10+ and its standard library. Run python3 main.py from any working directory. Optional --output path/report.json writes the JSON report, row/cluster CSV, first-two-feature SVG and a centroid/scaler JSON file. The SVG depicts only the first two fitted coordinates, so additional one-hot geometry is not fully visible there.

Source inputs are included in each role. The solution is a complete worked reference; compare its trace after attempting the named function.

Algorithm reference: https://scikit-learn.org/stable/modules/clustering.html

Role: starter.
