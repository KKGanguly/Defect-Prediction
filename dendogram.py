from math import ceil
import os
import numpy as np
import pandas as pd
from ezr_24Jun14.ezr import DATA, csv, dendogram, mids, where, showDendo
from timeit import default_timer as timer

class dendogramclassifer:
    def __init__(self, **params):
        self.set_params(**params)
        self.dendo = None
        self.stop = 12
    def set_params(self, **params): 
        pass

    def fit(self, x_train, y_train):
        #self.stop = ceil(len(x_train)/25)
        X = x_train
        X['timeopen!'] = y_train.astype(int)
        def row_gen():
            yield X.columns.tolist()
            for row in X.values.tolist():
                row[-1] = int(row[-1])
                yield row
        data1=DATA(row_gen())
        self.dendo = dendogram(data1,stop = self.stop)

        return self
    def predict(self,x_test):
        headers = x_test.columns.tolist()
        def row_gen():
            yield headers
            for row in x_test.values.tolist():
                yield row
        data1=DATA(row_gen())
        res = []
        for row in data1.rows:
            node =where(data1,self.dendo,row)
            res.append(str(mids(node,node.cols.y)[0]))
        return res