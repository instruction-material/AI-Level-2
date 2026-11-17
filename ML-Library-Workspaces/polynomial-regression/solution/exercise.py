def fit_model(features, targets, degree=2):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
    transform = PolynomialFeatures(degree=degree, include_bias=False)
    expanded = transform.fit_transform(features)
    return transform, LinearRegression().fit(expanded, targets)
