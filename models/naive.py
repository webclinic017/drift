from models.base import Model
import numpy as np

class StaticNaiveModel(Model):
    '''
    Model that carries the last observation (from returns) to the next one, naively.
    '''

    # data_format = 'dataframe'
    data_scaling = 'unscaled'
    only_column = None

    def fit(self, X, y):
        # This is a static model, it can' learn anything
        pass

    def predict(self, X):
        return np.array([X[-1][0]])

    def clone(self):
        return self