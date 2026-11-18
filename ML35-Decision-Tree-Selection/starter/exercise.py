"""Select depth from the supplied validation accuracies."""


def choose_depth(scores):
    """Return the shallowest depth within 0.02 of the highest accuracy.

    scores maps integer depths 0 through 8 to finite accuracies from 0 to 1.
    Validate this contract. Treat the 0.02 boundary as inclusive with 1e-12
    numerical tolerance. Use only this mapping; final-test labels are withheld.
    """
    raise NotImplementedError("Complete choose_depth(scores).")
