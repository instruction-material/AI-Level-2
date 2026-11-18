# Source and data

This authored course exercise reuses the complete bounded tree rules from
AI-Level-2 commit `609c914d179ffde8518cfbfdb74f6739f9a0418d`,
`ML-06-ml3-5-decision-trees-and-interpretable-models-supplemental-3/solution/main.py`.
That reference remains the separate recursive-growth implementation exercise.

The three CSV splits are constructed, fictional two-feature measurements.
The x coordinate determines the authored class rule; y alternates as a
distractor. They contain no personal data or measured external outcomes.

The tree supports two finite numeric features and depths 0 through 8.
Thresholds are midpoints of distinct training values. Route values
`<= threshold` left. Maximize positive Gini reduction, using feature
index then threshold for ties within `1e-12`. Leaves use training
class counts, with alphabetical class ties. The positive-gain rule can
stop on XOR even though a deeper tree could represent XOR.

[Algorithm context](https://scikit-learn.org/stable/modules/tree.html)
