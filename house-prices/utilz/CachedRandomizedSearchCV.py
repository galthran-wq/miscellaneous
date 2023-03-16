from joblib import load, dump
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import RandomizedSearchCV
import numpy as np
import pandas as pd

class CachedRandomizedSearchCV(BaseEstimator, TransformerMixin):
    """Proxt for RSCV with joblib caching"""
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        cache_file_name,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        self.cache_file_name = cache_file_name
        try:
            cache_file = open(self.cache_file_name, 'rb')
            self.rs_  = load(cache_file)
            cache_file.close()
        except FileNotFoundError:
            self.rs_ = RandomizedSearchCV(
                estimator, param_distributions,
                n_iter=n_iter, scoring=scoring, n_jobs=n_jobs,
                refit=refit, cv=cv, verbose=verbose,
                pre_dispatch=pre_dispatch,
                random_state=random_state,
                error_score=error_score,
                return_train_score=return_train_score
            )
    
    def fit(self, X, y=None):
        self.rs_.fit(X, y)
        cache_file =  open(self.cache_file_name, 'wb')
        dump(self.rs_, cache_file)
        return self
    
    def __getattr__(self, attr):
        return getattr(self.rs_, attr)