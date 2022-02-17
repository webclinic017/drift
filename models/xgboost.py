# from __future__ import annotations
# from models.base import Model
# import numpy as np
# from xgboost import XGBClassifier

# class XGBoostModel(XGBClassifier):

#     method = 'classification'
#     data_transformation = 'transformed'
#     only_column = None
#     predict_window_size = 'single_timestamp'

#     def fit(self, X: np.ndarray, y: np.ndarray) -> None:
#         def map_to_xgb(y): return np.array([1 if i == 1 else 0 for i in y])
#         self.fit(X, map_to_xgb(y))

#     def predict(self, X) -> tuple[float, np.ndarray]:
#         pred = self.predict(X).item()
#         probability = self.predict_proba(X).squeeze()
#         def map_from_xgb(y): return 1 if y == 1 else -1
#         return (map_from_xgb(pred), probability)
