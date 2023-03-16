from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class InteractionFeatures(BaseEstimator, TransformerMixin):
    """Proxy for PolynomialFeatures"""
    def __init__(
        self, degree=2, *, interaction_only=False, include_bias=True, order="C"
    ):
        self.pol_ = PolynomialFeatures(degree, interaction_only=interaction_only, include_bias=include_bias, order=order)
        self.main_effects_num = None
        
    def fit(self, X, y=None):
        self.main_effects_num = X.shape[1]
        
        if self.pol_.include_bias:
            self.main_effects_num +=1
            
        self.pol_.fit(X, y)
        return self
    
    def get_feature_names_out(self, *args, **kwargs):
        feats = self.pol_.get_feature_names_out()
        return feats[self.main_effects_num:]
    
    def transform(self, X):
        """
        Use the fact that transform returns lower order features first:
        1, a, b, a^2, ab, b^2
        """
        X_tr = self.pol_.transform(X)
        return X_tr[:, self.main_effects_num:]
    
    def __getattr__(self, attr):
        return getattr(self.pol_, attr)