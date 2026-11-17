def fit_classifier(features, targets, neighbors=3):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([
        ("scale", StandardScaler()),
        ("neighbors", KNeighborsClassifier(n_neighbors=neighbors, weights="uniform")),
    ]).fit(features, targets)
