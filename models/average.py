from models.base import Model
import numpy as np

class StaticAverageModel(Model):
    '''
    Model that averages .
    '''

    data_scaling = 'unscaled'
    only_column = 'model_'
    feature_selection = 'off'
    model_type = 'static'

    def fit(self, X, y, prev_model):
        # This is a static model, it can' learn anything
        pass

    def predict(self, X):
        # Make sure there's data to average
        assert X.shape[1] > 0
        prediction = np.average(X[0])
        return np.array([prediction])

    def clone(self):
        return self