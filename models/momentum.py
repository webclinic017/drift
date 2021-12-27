from models.base import Model
import numpy as np

class StaticMomentumModel(Model):
    '''
    Model that uses only one feature: momentum. It's positive if momentum is greater than 0, otherwise it's negative.
    '''

    data_scaling = 'unscaled'
    only_column = 'mom'
    feature_selection = 'off'
    model_type = 'static'

    def __init__(self, allow_short: bool) -> None:
        super().__init__()
        self.allow_short = allow_short

    def fit(self, X, y, prev_model):
        # This is a static model, it can' learn anything
        pass

    def predict(self, X):
        negative_class = -1.0 if self.allow_short == True else 0.0
        prediction = 1.0 if X[-1][0] > 0 else negative_class
        return np.array([prediction])

    def clone(self):
        return self