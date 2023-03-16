from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class CategoryShrinker(BaseEstimator, TransformerMixin):
    
    DOMINATED_RATIO= 0.95
    INFREQUENT_RATIO= 0.03
    
    def __init__(self, 
                 remove_dominated=True, 
                 combine_infrequent=True,
                 bin_years=True,
                ):
        self.remove_dominated = remove_dominated
        self.combine_infrequent = combine_infrequent
        self.bin_years = bin_years
        
        self.dominated_cols = []
        # col: [cat]
        self.infrequent_map = {}

    def fit(self, X, y=None):
        if self.remove_dominated:
           self.discover_dominated_(X)
        if self.combine_infrequent:
           self.discover_infrequent_(X)
        self.out_cols = X.columns.drop(self.dominated_cols)
        return self

    def transform(self, X):
        # order matters
        if self.bin_years:
            X = self.bin_years_(X)
        if self.remove_dominated:
            X = self.remove_dominated_(X)
        if self.combine_infrequent:
            X = self.combine_infrequent_(X)
        
        return X 
    
    def get_feature_names_out(self, *args, **kwargs):
        return self.out_cols
    
    # KBins doesn't support NaNs
    def bin_years_(self, X):
        res = X.copy()
        to_bin = ["YearBuilt", "GarageYrBlt", "YearRemodAdd"]
        for col in to_bin:
            res[col] = pd.cut(
                X["GarageYrBlt"],
                [1900, 1940, 1960, 1980, 2000, np.inf],
                labels=["1900-1940", "1940-1960", "1960-1980", "1980-2000", "2000+"]
            )
        return res
    
    def discover_dominated_(self, X):
        for col in X.columns:
            if (
                (X[col].value_counts() > X.shape[0] * self.DOMINATED_RATIO).any() | 
                X[col].isna().sum() > X.shape[0] * self.DOMINATED_RATIO 
            ):
                self.dominated_cols.append(col)
                
    def discover_infrequent_(self, X):
        for col in X.columns:
            values = X[col].value_counts()
            infrequent_cats = values[values < X.shape[0] * self.INFREQUENT_RATIO].index
            if len(infrequent_cats) > 1:
                self.infrequent_map[col] = infrequent_cats
 
    def remove_dominated_(self, X):
        return X.drop(columns=self.dominated_cols)
    
    def combine_infrequent_(self, X):
        """
        If a certain category doesn't cover {infrequent_ratio}% of entries, comb 
        """
        res = X.copy()
        placeholder = "Other"
        
        for col in self.infrequent_map:
            if col in res:
                res[col].replace({
                    infrequent_cat:placeholder 
                    for infrequent_cat in self.infrequent_map[col]
                }, inplace=True)
        return res