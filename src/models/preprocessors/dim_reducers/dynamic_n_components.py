from sklearn.base import BaseEstimator, TransformerMixin

class DynamicDimensionReducer(BaseEstimator, TransformerMixin):
    """
    Handles dynamic dimensionality reduction by adjusting the number of components
    in the provided estimator. This class modifies the number of components (`n_components`)
    in the `estimator` based on the number of features in the input data and optimizes it
    for sensible values. The adjustment ensures that `n_components` is at least 1 and does not
    exceed the number of features minus one.

    :ivar estimator: The dimensionality reduction estimator provided by the user.
        This is expected to implement the `fit` and `transform` methods for compatibility.
    :type estimator: Any
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None):
        # Determine the appropriate n_components based on the number of features in X
        n_features = X.shape[1]
        self.estimator.n_components = max(1, min(n_features - 1, self.estimator.n_components))
        self.estimator.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.estimator.transform(X)