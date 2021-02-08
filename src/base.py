class BaseModel:
    def fit(self, X, y=None, *args, **kwargs):
        raise NotImplementedError("fit method is not implemented yet")

    def transform(self, X):
        raise NotImplementedError("transform method is not implemented yet")

    def fit_transform(self, X, y=None, *args, **kwargs):
        raise NotImplementedError("fit_transform method is not implemented yet")

    def predict(self, X):
        raise NotImplementedError("predict method is not implemented yet")
