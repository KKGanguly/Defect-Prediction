import os
import numpy as np
import pandas as pd
from ezr_24Jun14.ezr import DATA, csv, dendogram, mids, where, showDendo
from dendogram import dendogramclassifer

class randomforestclassifer:
    def __init__(self, **params):
        self.set_params(**params)
        self.trees = []
        self.max_tree_nums = 10
    
    #bootstrapping may introduce duplicates which may cause dendogram to fail
    def subsample(self, X, y, ratio=1.0):
        n_samples = int(X.shape[0] * ratio)
        indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
        return X.iloc[indices], y.iloc[indices]

    def set_params(self, **params): 
        pass
    
    def fit(self, x_train, y_train):
        y_train = pd.DataFrame(data = y_train, columns=["timeopen!"])        
        y_train['timeopen!'] = y_train['timeopen!'].astype(int)
        self.trees = []
        for i in range(self.max_tree_nums):
            X, y = self.subsample(x_train,y_train,0.7)
            dendo = dendogramclassifer()
            self.trees.append(dendo.fit(X,y))
        return self
    def most_common_label(self,y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]
    def predict(self,x_test):
        tree_preds = np.array([tree.predict(x_test) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [self.most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)