# Source Backlog Ledger

Current status: no unlinked source folders remain at the active top level.

On 2026-05-14, 4 top-level folders that were not linked by the live course catalog were moved to `_archived-unlinked/`. That archive is retained for source-history review, optional-bank recovery, or future promotion, but it is not part of the active public course surface.

Promotion rule: restore a folder from `_archived-unlinked/` only after the live course text names where it belongs, whether it is starter or solution material, and how the student or tutor verifies it. Update `COURSE_SOURCE_MANIFEST.md` and the live catalog link in the same change.

## Supplemental algorithm review, 2026-11-16

All 30 generated supplemental reference packs originally repeated CSV count/total/mean
statistics. The first eight starter/reference pairs now teach their named methods:
training-only scaled KNN, weighted KNN, Gaussian Naive Bayes and Multinomial Naive
Bayes, a Gini decision stump and a bounded recursive decision tree, a sigmoid neuron and a hidden-layer network. The two KNN pairs have 20 independent checks; the two Naive Bayes pairs
have 19; the two decision-tree pairs have 24; the neural-network pairs have 24. Each repaired role includes its own fixed training/test data and a
complete exercise with hand-worked expected results.

The remaining 22 generated pairs still need algorithm-specific source repair and
verification. This inventory gate verifies source presence; it does not certify
algorithm correctness or complete course/browser acceptance.
