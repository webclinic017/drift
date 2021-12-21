from models.base import Model
import numpy as np

class StaticAverageModel(Model):
    '''
    Model that averages .
    '''

    # data_format = 'dataframe'
    data_scaling = 'unscaled'
    only_column = 'model_'

    def fit(self, X, y):
        # This is a static model, it can' learn anything
        pass

    def predict(self, X):
        # Make sure there's data to average
        assert X.shape[1] > 0
        prediction = np.average(X[0])
        return np.array([prediction])

    def clone(self):
        return self