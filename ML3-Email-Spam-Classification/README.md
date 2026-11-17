# Fictional email classification

The supplied `fictional-emails.csv` contains 64 authored messages, half ham (0) and half spam (1), plus two deliberately repeated rows. It contains no inbox exports. The original 5,171-row `emails.csv` remains a historical dataset and is unused by this exercise.

Use Python 3.14 in an isolated environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
.venv/bin/python main.py --output-dir output
```

The original named reference script can also run directly. For the notebooks, open the notebook in the folder containing `fictional-emails.csv`, select this environment's Python kernel, and run the cells from the beginning. Their actual program is embedded locally. The normal run uses the supplied file and installed packages; it downloads neither messages nor language resources. Output is `spam-report.json` plus a concise console report with no message text.

## Training boundary and named task

Complete `fit_pipeline(messages, targets, binary=False)` in the starter. Return a fitted `Pipeline` containing default `CountVectorizer(binary=binary)` and `MultinomialNB(alpha=1.0)`, fitted only on the supplied training messages and labels. Keep the returned pipeline so that later `predict()` applies the same fitted vocabulary.

The scaffold removes duplicate text/label pairs, rejects contradictory labels, then stratifies a fixed seed-42 split into 38 training, 13 validation and 13 final test rows. Raw text is split before any learned vocabulary. The default vectorizer lowercases and counts tokens of at least two word characters; it has no downloaded stopword list. Compare one documented change, binary presence instead of counts, on validation rows. The default count model is fixed before opening final test results; this exercise does not tune on test scores.

Compare final accuracy, class precision/recall/F1, macro F1 and confusion matrix with a most-frequent dummy fitted on training labels. Rows of the matrix are actual labels and columns predictions, ordered ham=0, spam=1. Inspect any reported fictional row IDs locally. Precision asks how many predicted spam messages are spam; recall asks how many actual spam messages were found. Accuracy alone can hide poor recall when spam is rare. The fixture's artificial class balance differs from a real inbox.

## Worked count model

For just two training messages, ham `meeting meeting note` and spam `prize prize note`, the vocabulary is meeting/note/prize. With alpha=1 smoothing, ham probabilities are 3/6, 2/6, 1/6; spam probabilities are 1/6, 2/6, 3/6. Both class priors are 1/2. For `meeting note`, the unnormalized ham score is `(1/2)(3/6)(2/6)=1/12`, and spam is `(1/2)(1/6)(2/6)=1/36`. Their normalized probabilities are 3/4 and 1/4. Implementations add logarithms to avoid multiplying many tiny values. A word absent from the training vocabulary contributes no count, even if it appears in a held-out message.

For actual labels 0,0,1 and predictions 0,1,1, the matrix is `[[1,1],[0,1]]`; spam precision is 1/2, recall 1 and F1 2/3. These are hand checks, separate from the larger fixture's measured result.

Repeated template words make this authored fixture deliberately easy. Its measured scores establish that this particular pipeline runs and preserves the split, not that it detects arbitrary spam. Word-count Naive Bayes also ignores word order and treats features as conditionally independent given a class. Keep these limits and the saved split, vocabulary, predictions and versions with the result.

The historical NLTK/remote-data program remains available in [the prior revision](https://github.com/instruction-material/AI-Level-2/tree/fbe141d92a663593d0958021b662c2d50863cd65/ML3-Email-Spam-Classification). Its whole-table vocabulary fit precedes its split; that is the leakage corrected here.

Role: completed reference.
