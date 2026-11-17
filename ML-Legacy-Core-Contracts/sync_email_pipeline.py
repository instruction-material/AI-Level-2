"""Synchronize original scripts, notebooks and a separate standalone starter role."""
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPLETE = Path(__file__).with_name("email_pipeline.py").read_text()
BEGIN, END = COMPLETE.index("def fit_pipeline("), COMPLETE.index("\n\ndef metrics(")
STARTER = COMPLETE[:BEGIN] + '''def fit_pipeline(messages, targets, binary=False):
    """Fit CountVectorizer and MultinomialNB together on training messages only."""
    raise NotImplementedError("Complete fit_pipeline using the supplied training messages and targets")
''' + COMPLETE[END:]
words = "amber birch cedar dune elm fern grove harbor iris jade kite lake maple north oak pine quartz river stone tide umber vale willow xenon yarrow zinnia acorn brook cloud delta echo frost".split()
rows = []
for index, word in enumerate(words):
    rows += [[f"ham-{index:02}", f"Project meeting schedule note {word} agenda tomorrow", 0],
             [f"spam-{index:02}", f"Prize offer reward claim {word} coupon today", 1]]
rows += [rows[0], rows[-1]]
buffer = io.StringIO(newline="")
writer = csv.writer(buffer, lineterminator="\n")
writer.writerow(["message_id", "text", "label_num"])
writer.writerows(rows)
FIXTURE = buffer.getvalue()
README = """# Fictional email classification

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
"""


def notebook(program, filename, original=None):
    nb = json.loads(original.read_text()) if original and original.exists() else {"metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    nb["nbformat"], nb["nbformat_minor"] = 4, 5
    definitions = program.rsplit('\n\nif __name__ == "__main__":', 1)[0] + "\n"
    cell = ("from pathlib import Path\nimport sys\n"
            + "__file__ = str(Path.cwd() / " + repr(filename) + ")\n"
            + "_previous_argv = sys.argv[:]\nsys.argv = [__file__, '--output-dir', 'output']\ntry:\n"
            + "    main()\n"
            + "finally:\n    sys.argv = _previous_argv\n")
    nb["cells"] = [{"cell_type": "markdown", "id": "email-local-instructions", "metadata": {},
                    "source": ["# Fictional email classification\n", "Use the Python environment and local fixture described in README.md. Run from the folder containing fictional-emails.csv.\n"]},
                   {"cell_type": "code", "id": "email-training-pipeline", "metadata": {},
                    "source": definitions.splitlines(keepends=True), "outputs": [], "execution_count": None},
                   {"cell_type": "code", "id": "email-run-experiment", "metadata": {},
                    "source": cell.splitlines(keepends=True), "outputs": [], "execution_count": None}]
    nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


reference = ROOT / "ML3-Email-Spam-Classification"
starter = ROOT / "ML3-Email-Spam-Classification-Starter"
starter.mkdir(exist_ok=True)
for folder, program in [(reference, COMPLETE), (starter, STARTER)]:
    (folder / "main.py").write_text(program)
    (folder / "fictional-emails.csv").write_text(FIXTURE)
    (folder / "requirements.txt").write_bytes((ROOT / "requirements-ml-legacy.txt").read_bytes())
    (folder / "README.md").write_text(README + ("\nRole: named starter.\n" if folder == starter else "\nRole: completed reference.\n"))
(reference / "Email-Spam-Classifier.py").write_text(COMPLETE)
(reference / "Email-Spam-Classifier-Starter.py").write_text(STARTER)
for name, program, filename in [("EmailSpamClassifier.ipynb", COMPLETE, "Email-Spam-Classifier.py"),
                                ("EmailSpamClassifierStarter.ipynb", STARTER, "Email-Spam-Classifier-Starter.py")]:
    path = reference / name
    path.write_text(json.dumps(notebook(program, filename, path), indent=2) + "\n")
