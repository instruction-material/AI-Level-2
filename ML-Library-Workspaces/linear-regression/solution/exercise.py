def fit_model(features, targets):
    from sklearn.linear_model import LinearRegression
    return LinearRegression().fit(features, targets)
