from models.sklearn import SKLearnModel
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.ensemble import RandomForestRegressor
from .base import Model

default_feature_selector_classification = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
default_feature_selector_regression = SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1), 'regression')

def get_model(model_name: str) -> Model:

    if model_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        return SKLearnModel(LinearRegression(n_jobs=-1), 'regression')
    elif model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        return SKLearnModel(Lasso(alpha=100, random_state=1), 'regression')
    elif model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        return SKLearnModel(Ridge(alpha=0.1), 'regression')
    elif model_name == 'BayesianRidge':
        from sklearn.linear_model import BayesianRidge
        return SKLearnModel(BayesianRidge(), 'regression')
    elif model_name == 'KNN':
        from sklearnex.neighbors import KNeighborsRegressor
        return SKLearnModel(KNeighborsRegressor(n_neighbors=25), 'regression')
    elif model_name == 'AB':
        from sklearn.ensemble import AdaBoostRegressor
        return SKLearnModel(AdaBoostRegressor(random_state=1), 'regression')
    elif model_name == 'MLP':
        from sklearn.neural_network import MLPRegressor
        return SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000), 'regression')
    elif model_name == 'RFR':
        return SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1), 'regression')
    elif model_name == 'SVR':
        from sklearnex.svm import SVR
        return SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1), 'regression')
    elif model_name == 'StaticNaive':
        from models.naive import StaticNaiveModel
        return StaticNaiveModel()
    elif model_name == 'DNN':
        from models.neural import LightningNeuralNetModel
        from models.pytorch.neural_nets import MultiLayerPerceptron
        import torch.nn.functional as F
        return LightningNeuralNetModel(
            MultiLayerPerceptron(
                hidden_layers_ratio = [1.0], 
                probabilities = False, 
                loss_function = F.mse_loss), 
            max_epochs=15
        )

    elif model_name == 'LogisticRegression_two_class':
        from sklearn.linear_model import LogisticRegression
        return SKLearnModel(LogisticRegression(C=10, random_state=1, solver='liblinear', max_iter=1000), 'classification')
    elif model_name == 'LogisticRegression_three_class':
        from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX
        return SKLearnModel(LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1), 'classification')
    elif model_name == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return SKLearnModel(LinearDiscriminantAnalysis(), 'classification')
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        return SKLearnModel(KNeighborsClassifier(), 'classification')
    elif model_name == 'CART':
        from sklearn.tree import DecisionTreeClassifier
        return SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1), 'classification')
    elif model_name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        return SKLearnModel(GaussianNB(), 'classification')
    elif model_name == 'AB':
        from sklearn.ensemble import AdaBoostClassifier
        return SKLearnModel(AdaBoostClassifier(n_estimators=15), 'classification')
    elif model_name == 'RFC':
        return SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
    elif model_name == 'SVC':
        from sklearn.svm import SVC
        return SKLearnModel(SVC(kernel='rbf', C=1e3, probability=True, random_state=1), 'classification')
    elif model_name == 'XGB_two_class':
        from xgboost import XGBClassifier
        from models.xgboost import XGBoostModel
        return XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss'))
    elif model_name == 'LGBM':
        from lightgbm import LGBMClassifier
        return SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
    elif model_name == 'StaticMom':
        from models.momentum import StaticMomentumModel
        return StaticMomentumModel(allow_short=True)
    elif model_name == 'Average':
        from models.average import StaticAverageModel
        return StaticAverageModel()
    else:
        raise Exception(f'Model {model_name} not found')