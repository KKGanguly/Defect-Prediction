from math import ceil
from sklearn.tree import DecisionTreeClassifier


class CustomDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, min_samples_split=2, **kwargs):
        super().__init__(min_samples_split=min_samples_split, **kwargs)
    
    def fit(self, X, y, sample_weight=None, check_input=True):
        self.min_samples_split = ceil(len(X)/25)
        
        return super().fit(X, y, sample_weight=sample_weight,
                           check_input=check_input)