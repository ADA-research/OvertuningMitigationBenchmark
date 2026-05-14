from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, VarianceThreshold


class KeepNFeaturesWrapper(BaseEstimator, TransformerMixin):
    """
    Wrapper that ensures at least a minimum number of features are kept.
    If the wrapped estimator would select fewer features, uses SelectKBest instead.
    """

    def __init__(self, estimator, threshold):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X, y=None):
        if (X.var(axis=0) > self.threshold).astype(int).sum() < 5:
            self.estimator = SelectKBest(
                score_func=lambda _X, _y=None: _X.var(axis=0),
                k=5
            )
        else:
            self.estimator = VarianceThreshold(
                threshold=self.threshold
            )

        self.estimator.fit(X, y)
        return self

    def transform(self, X):
        return self.estimator.transform(X)
