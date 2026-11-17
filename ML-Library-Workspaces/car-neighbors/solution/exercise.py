FEATURES = ['buying', 'maintenance', 'doors', 'persons', 'lug boot', 'safety']

def fit_classifier(features, targets, neighbors=7):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.neighbors import KNeighborsClassifier
    pipeline = Pipeline([
        ("encode", ColumnTransformer([("categories", OneHotEncoder(handle_unknown="ignore", sparse_output=False), FEATURES)], remainder="drop")),
        ("neighbors", KNeighborsClassifier(n_neighbors=neighbors, weights="uniform")),
    ])
    pipeline.fit(features, targets)
    return pipeline
