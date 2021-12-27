from models.base import Model
import numpy as np

class StaticNaiveModel(Model):
    '''
    Model that carries the last observation (from returns) to the next one, naively.
    '''

    data_scaling = 'unscaled'
    only_column = None
    feature_selection = 'off'
    model_type = 'static'

    def fit(self, X, y, prev_model):
        # This is a static model, it can' learn anything
        pass

    def predict(self, X):
        return np.array([X[-1][0]])

    def clone(self):
        return self