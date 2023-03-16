from itertools import chain
from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer


class PandasTransfomerWrapper(BaseEstimator, TransformerMixin):
    """A wrapper around any transformer to facilitate
    recovery of column (feature) names"""

    def __init__(self, transformer, *args, **kwargs):
        self.transformer_ = transformer(*args, **kwargs)
        self.transformed_col_names = []

    def fit(self, X: pd.DataFrame, y=None):
        """Fit transformer_, and obtain names of transformed columns in advance

        Args:
            X (pd.DataFrame): DataFrame to be fitted on
            y (Any, optional): Purely for compliance with transformer API. Defaults to None.
        """
        assert isinstance(X, pd.DataFrame)
        self.col_transformer = self.transformer_.fit(X)
        self.transformed_col_names = self.transformer_.get_feature_names_out()
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform a new DataFrame using fitted self.transformer_

        Args:
            X (pd.DataFrame): DataFrame to be transformed

        Returns:
            pd.DataFrame: DataFrame transformed by self.transformer_
        """
        assert isinstance(X, pd.DataFrame)
        transformed_X = self.transformer_.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(transformed_X, index=X.index, 
            columns=self.transformed_col_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                transformed_X, index=X.index,
                columns=self.transformed_col_names
            )
