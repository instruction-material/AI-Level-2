def fit_pipeline(messages, targets, binary=False):
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
    pipeline = Pipeline([("words", CountVectorizer(binary=binary)),
                         ("classifier", MultinomialNB(alpha=1.0))])
    pipeline.fit(messages, targets)
    return pipeline
