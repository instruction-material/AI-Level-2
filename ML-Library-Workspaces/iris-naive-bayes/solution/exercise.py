def fit_classifier(features, targets):
    from sklearn.naive_bayes import GaussianNB
    return GaussianNB(var_smoothing=1e-9).fit(features, targets)
