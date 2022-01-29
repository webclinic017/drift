from models.sklearn import SKLearnModel
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.ensemble import RandomForestRegressor
from .base import Model

default_feature_selector_classification = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')

def get_model(model_name: str) -> Model:

    def set_name(model: Model) -> Model:
        model.name = model_name
        return model

    if model_name == 'LinearRegression':
        from sklearn.linear_model import LinearRegression
        return set_name(SKLearnModel(LinearRegression(n_jobs=-1), 'regression'))
    elif model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        return set_name(SKLearnModel(Lasso(alpha=100, random_state=1), 'regression'))
    elif model_name == 'Ridge':
        from sklearn.linear_model import Ridge
        return set_name(SKLearnModel(Ridge(alpha=0.1), 'regression'))
    elif model_name == 'BayesianRidge':
        from sklearn.linear_model import BayesianRidge
        return set_name(SKLearnModel(BayesianRidge(), 'regression'))
    elif model_name == 'KNN':
        from sklearnex.neighbors import KNeighborsRegressor
        return set_name(SKLearnModel(KNeighborsRegressor(n_neighbors=25), 'regression'))
    elif model_name == 'AB':
        from sklearn.ensemble import AdaBoostRegressor
        return set_name(SKLearnModel(AdaBoostRegressor(random_state=1), 'regression'))
    elif model_name == 'MLP':
        from sklearn.neural_network import MLPRegressor
        return set_name(SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000), 'regression'))
    elif model_name == 'RFR':
        return set_name(SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1), 'regression'))
    elif model_name == 'SVR':
        from sklearnex.svm import SVR
        return set_name(SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1), 'regression'))
    elif model_name == 'StaticNaive':
        from models.naive import StaticNaiveModel
        return set_name(StaticNaiveModel())
    elif model_name == 'DNN':
        from models.neural import LightningNeuralNetModel
        from models.pytorch.neural_nets import MultiLayerPerceptron
        import torch.nn.functional as F
        return set_name(LightningNeuralNetModel(
            MultiLayerPerceptron(
                hidden_layers_ratio = [1.0], 
                probabilities = False, 
                loss_function = F.mse_loss), 
            max_epochs=15
        ))

    elif model_name == 'LogisticRegression_two_class':
        from sklearn.linear_model import LogisticRegression
        return set_name(SKLearnModel(LogisticRegression(C=10, random_state=1, solver='liblinear', max_iter=1000), 'classification'))
    elif model_name == 'LogisticRegression_three_class':
        from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX
        return set_name(SKLearnModel(LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1), 'classification'))
    elif model_name == 'LDA':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return set_name(SKLearnModel(LinearDiscriminantAnalysis(), 'classification'))
    elif model_name == 'KNN':
        from sklearn.neighbors import KNeighborsClassifier
        return set_name(SKLearnModel(KNeighborsClassifier(), 'classification'))
    elif model_name == 'CART':
        from sklearn.tree import DecisionTreeClassifier
        return set_name(SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1), 'classification'))
    elif model_name == 'NB':
        from sklearn.naive_bayes import GaussianNB
        return set_name(SKLearnModel(GaussianNB(), 'classification'))
    elif model_name == 'AB':
        from sklearn.ensemble import AdaBoostClassifier
        return set_name(SKLearnModel(AdaBoostClassifier(n_estimators=15), 'classification'))
    elif model_name == 'RFC':
        return set_name(SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification'))
    elif model_name == 'SVC':
        from sklearn.svm import SVC
        return set_name(SKLearnModel(SVC(kernel='rbf', C=1e3, probability=True, random_state=1), 'classification'))
    elif model_name == 'XGB_two_class':
        from xgboost import XGBClassifier
        from models.xgboost import XGBoostModel
        return set_name(XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss')))
    elif model_name == 'LGBM':
        from lightgbm import LGBMClassifier
        return set_name(SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification'))
    elif model_name == 'StaticMom':
        from models.momentum import StaticMomentumModel
        return set_name(StaticMomentumModel(allow_short=True))
    elif model_name == 'Average':
        from models.average import StaticAverageModel
        return set_name(StaticAverageModel())
    else:
        raise Exception(f'Model {model_name} not found')