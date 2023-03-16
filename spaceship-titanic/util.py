import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.impute import SimpleImputer
import math


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, grid, pred_preprocess_hook: function = lambda x : x) -> None:
        super().__init__()
        self.model = model
        self.grid = grid
        self.models = [ self.model(**params) for params in grid ]
    
    def fit(self, X, y, *args, **kwargs):
        for model in self.models:
            model.fit(X, y, *args, **kwargs)
        return self
    
    def predict(self, X):
        preds = np.zeros(len(X))
        for model in self.models:
            preds += self.pred_preprocess_hook(model.predict(X))
        preds /= len(self.models)
        # majority vote
        return np.where(preds > 0.5, 1, 0)


class IsNanMixin:
    NONE_CAT = "none"
    NONE_NUM = 0

    def isnan(self, cabin_val):
        """
        I'm not sure why None values are parsed as floats
        A more obvious choice would be ```is None```
        """
        return (
            isinstance(cabin_val, float) and math.isnan(cabin_val) or 
            cabin_val == self.NONE_CAT or
            cabin_val is None
        )


class CabinExtractor(IsNanMixin, TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self.cabins = None 
        self.num_imputer = SimpleImputer(strategy="median")
    
    def fit(self, X: pd.Series, y=None):
        assert len(X.shape) == 1
        X_parsed = X.map(lambda x: x.split("/") if not self.isnan(x) else CabinExtractor.NONE_CAT)
        deck = X_parsed.map(lambda x: x[0] if not self.isnan(x) else CabinExtractor.NONE_CAT)
        num = X_parsed.map(lambda x: x[1] if not self.isnan(x) else CabinExtractor.NONE_CAT)
        side = X_parsed.map(lambda x: x[2] if not self.isnan(x) else CabinExtractor.NONE_CAT)
        self.decks = set(deck.values)
        self.nums = set(num.values)
        self.sides = set(side.values)
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        X = X.map(lambda x: x.split("/") if not self.isnan(x) else CabinExtractor.NONE_CAT)
        num = self.num_imputer.fit_transform(
            X.apply(lambda x: int(x[1]) if not self.isnan(x) and x[1] in self.nums else None).to_frame()
        )[:, 0]
        return pd.DataFrame({
            "deck": X.apply(lambda x: x[0] if not self.isnan(x) and x[0] in self.decks else self.NONE_CAT),
            "side": X.apply(lambda x: x[2] if not self.isnan(x) and x[2] in self.sides else self.NONE_CAT),
            "num": num,
        })


# class GenderExtractor(IsNanMixin, TransformerMixin, BaseEstimator):
#     def __init__(self) -> None:
#         super().__init__()
#         self._detector = gender.Detector()
    
#     def detect(self, x):
#         if isinstance(x, str):
#             x = x.split(" ")[0]
#         pred = self._detector.get_gender(x)
#         if pred == "andy" or pred == "unknown":
#             pred = self.NONE_CAT
#         return pred
    
#     def fit(self, X: pd.Series, y=None):
#         return self
    
#     def transform(self, X: pd.Series) -> pd.DataFrame:
#         return X.apply(self.detect).to_frame()


class ExtractGroupMembership(TransformerMixin, BaseEstimator):
    def __init__(self) -> None:
        super().__init__()
        self._extract_group = lambda x: x[:4]
        self._extract_relpos = lambda x: x[-2:]
        self.group_to_size = None

    def fit(self, X: pd.Series, y=None):
        return self
    
    def transform(self, X: pd.Series) -> pd.DataFrame:
        assert len(X.shape) == 1
        group = X.map(self._extract_group)
        relpos = X.map(self._extract_relpos)
        self.group_to_size = dict(group.groupby(group).count())
        return pd.DataFrame({
            "group_size": group.map(lambda x: self.group_to_size.get(x, 1)),
            "relpos": relpos,
        })
        # return group.map(lambda x: self.group_to_size.get(x, 1)).to_frame()


class FunctionalTransformer(IsNanMixin, TransformerMixin, BaseEstimator):
    def __init__(self, f) -> None:
        super().__init__()
        self.f = f 
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, x: pd.DataFrame):
        return x.apply(lambda series: series.map(lambda entry: self.f(entry) if not self.isnan(entry) else self.NONE_NUM))