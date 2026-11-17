# ML1-Customer-Segmentation-Updated

This solution uses manual centroid updates with a scikit-learn comparison on the CSV beside main.py.

Trace and explain the complete worked reference before using it to check a starter.

For a local run, create a Python 3.14 virtual environment and install this
folder's requirements.txt with python -m pip install --require-hashes -r
requirements.txt. Run python main.py from any directory. It writes before_clustering.png,
clustered_graph.png and clustering-report.json into the current directory;
--output-dir selects a different output directory.

Compare both reported k values across seeds 7, 19 and 31. Distances and inertia
use standardized features; plots and reported raw centroids retain the original
units. Cluster numbers can permute between runs. Read the inertia, silhouette,
and label-permutation-aware agreement together, then explain one uncertainty.
The displayed k is an example, not a statistically proved best number of groups.

Customer labels describe the supplied observations, not real customer types.
