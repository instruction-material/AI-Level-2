# Course Source Manifest

Canonical source repository: `AI-Level-2`

## Mapped Catalog Courses

- `machine-learning`: Machine Learning

## Verification Gate

- Run `./verify-course-source.sh` from this repository root before treating the source pack as ready.
- The verification gate checks for this manifest, the source backlog ledger, source-like files, removed Replit metadata, and any repo-specific readiness files.
- Project-specific unit tests or build commands should still be run inside individual project folders when a project includes its own test harness.

## Active Catalog Targets

| Folder |
| --- |
| `ML-01-ml2-k-nearest-neighbors-supplemental-2` |
| `ML-02-ml2-k-nearest-neighbors-supplemental-3` |
| `ML-03-ml3-naive-bayes-supplemental-2` |
| `ML-04-ml3-naive-bayes-supplemental-3` |
| `ML-05-ml3-5-decision-trees-and-interpretable-models-supplemental-2` |
| `ML-06-ml3-5-decision-trees-and-interpretable-models-supplemental-3` |
| `ML-07-ml4-neural-networks-supplemental-2` |
| `ML-08-ml4-neural-networks-supplemental-3` |
| `ML-09-ml5-introduction-to-regression-supplemental-2` |
| `ML-10-ml5-introduction-to-regression-supplemental-3` |
| `ML-11-ml6-regression-with-neural-networks-supplemental-2` |
| `ML-12-ml6-regression-with-neural-networks-supplemental-3` |
| `ML-13-ml7-image-classifier-supplemental-2` |
| `ML-14-ml7-image-classifier-supplemental-3` |
| `ML-15-ml7-5-model-evaluation-comparison-and-dataset-strategy-supplemental-2` |
| `ML-16-ml7-5-model-evaluation-comparison-and-dataset-strategy-supplemental-3` |
| `ML-17-ml8-master-project-supplemental-2` |
| `ML-18-ml8-master-project-supplemental-3` |
| `ML-19-applied-studio-12-customer-segmentation-supplemental-2` |
| `ML-20-applied-studio-12-customer-segmentation-supplemental-3` |
| `ML-21-applied-studio-13-customer-segmentation-interview-supplemental-2` |
| `ML-22-applied-studio-13-customer-segmentation-interview-supplemental-3` |
| `ML-23-applied-studio-14-customer-segmentation-supplemental-2` |
| `ML-24-applied-studio-14-customer-segmentation-supplemental-3` |
| `ML-25-applied-studio-15-disney-movie-clustering-supplemental-2` |
| `ML-26-applied-studio-15-disney-movie-clustering-supplemental-3` |
| `ML-27-applied-studio-16-disney-movie-clustering-supplemental-2` |
| `ML-28-applied-studio-16-disney-movie-clustering-supplemental-3` |
| `ML-29-applied-studio-17-knn-car-classification-supplemental-2` |
| `ML-30-applied-studio-17-knn-car-classification-supplemental-3` |
| `ML-Simple-Linear-Regression-Updated` |
| `ML-Simple-Polynomial-Regression-Starter-Updated` |
| `ML-Simple-Polynomial-Regression-Updated` |
| `ML1-Customer-Segmentation` |
| `ML1-Customer-Segmentation-Interview-Starter` |
| `ML1-Customer-Segmentation-Starter` |
| `ML1-Customer-Segmentation-Starter-Updated` |
| `ML1-Customer-Segmentation-Updated` |
| `ML1-Disney-Movie-Clustering` |
| `ML1-Disney-Movie-Clustering-Starter` |
| `ML1-Disney-Movie-Clustering-Starter-Updated` |
| `ML1-Disney-Movie-Clustering-Updated` |
| `ML1-Flower-Clustering` |
| `ML1-Flower-Clustering-Starter` |
| `ML2-KNN-Car-Classification` |
| `ML2-KNN-Car-Classification-Starter` |
| `ML2-KNN-Car-Classification-Updated` |
| `ML2-KNN-Customer-Segmentation-Classification` |
| `ML2-KNN-Customer-Segmentation-Classification-Updated` |
| `ML3-Email-Spam-Classification` |
| `ML3-Naive-Bayes-Iris-Flowers-Classification` |
| `ML4-Diabetes-Diagnosis-With-Neural-Networks` |
| `ML4-Neuron-Implementation` |
| `ML4-Simple-Neural-Network` |
| `ML5-Predicting-Life-Expectancy` |
| `ML5-Simple-Linear-Regression` |
| `ML5-Simple-Linear-Regression-Starter` |
| `ML6-Predicting-House-Prices` |
| `ML8-Master-Project` |
| `ML-Offline-Case-Fixtures` |
| `ML0-Offline-Data-Card` |
| `ML6-Offline-Housing-Comparison` |
| `ML7-5-Offline-Evaluation-Contract` |

## Source Inventory

- Active top-level course and fixture folders: 63
- Top-level archive container: 1
- Retained folders inside the archive: 4
- Source-like files: 192

## Scoped algorithm checks

Run python3 -B test_offline_housing.py -v for the three supplied fixtures,
the data-card/evaluation-contract cores, the offline housing comparison and
supplemental pairs 11 and 12. The first ten repaired supplemental folders
also contain their own independent unittest files. There are 145 checks
across this scoped suite. The inventory gate alone certifies source presence;
full course and hosted IDE acceptance remain separate.
