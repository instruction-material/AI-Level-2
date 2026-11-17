def fit_model(features, targets, degree):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    if degree not in (1, 2):
        raise ValueError("Choose the declared linear or quadratic comparison")
    return Pipeline([
        ("fill", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regression", LinearRegression() if degree == 1 else Ridge(alpha=1.0)),
    ]).fit(features, targets)
