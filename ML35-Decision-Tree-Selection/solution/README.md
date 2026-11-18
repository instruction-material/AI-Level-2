# Choose a shallow tree before final evaluation

Use Python 3.10 or later and its standard library. Keep `main.py`,
`exercise.py`, `tree_rules.py` and the `data` folder together:

```sh
python3 main.py --output tree-report.json
```

From another folder, use the absolute path to `main.py`; CSV paths remain
relative to the script. The output path names the report to save.

Complete `choose_depth(scores)` in the starter's `exercise.py`. The
reference completes this task. Select the shallowest candidate whose
validation accuracy is within 0.02 of the best, inclusively with `1e-12`
numerical tolerance. Validate a nonempty dictionary of finite accuracies
from 0 to 1 and integer depths from 0 through 8. Tree construction and
reporting are supplied.

## Worked comparison

The training rows have `x=0` through `7`, alternating `y=0,1`,
and labels A,A,B,B,B,B,A,A. Validation x values 0.5,2.5,4.5,6.5 have labels
A,B,B,A. Depths 0,1,2 score 0.5,0.75,1.0 on validation, selecting depth 2.
A separate map `{1:0.98,2:1.0}` selects depth 1 under the declared
simplicity rule.

The depth-2 tree asks `x<=1.5` at the root, then `x<=5.5` in its
right child. Draw the tree and training counts. Trace test x values
-0.5,1.5,5.5,7.5: predictions A,A,B,A, accuracy 1.0. Training class counts
tie, so the baseline selects A; it scores 0.75 on those same four test rows.

Keep the validation comparison and freeze the chosen fitted tree before
opening test labels. The script fits training rows alone. Changing final-test
labels can change reported accuracy while preserving candidate scores,
chosen depth, fitted counts and predictions.

## Fixed-model probe

Trace `[5.5,0]` and a copy `[5.51,0]`. The first predicts B and the
second A; only the second comparison changes direction. Preserve any
declared true label separately. This measures boundary sensitivity for these
queries, with no new fitting or model selection.

Record the candidate comparison, chosen depth, baseline, one complete path
and the saved JSON report. The tiny fictional fixture demonstrates the
mechanism; its perfect score supplies no evidence about an external population.
