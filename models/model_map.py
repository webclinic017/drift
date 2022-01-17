from models.sklearn import SKLearnModel
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.ensemble import RandomForestRegressor

default_feature_selector_classification = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
default_feature_selector_regression = SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1), 'regression')

def get_model_map(config:dict):

    model_map = {
        "primary_models": dict(),
        "ensemble_models": dict(),
    }
    
    combined_list = config['primary_models'] + config['meta_labeling_models'] + [config['ensemble_model']]
    for model_name in combined_list:
        if model_name == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            model_map['primary_models']['LR'] = SKLearnModel(LinearRegression(n_jobs=-1), 'regression')
        elif model_name == 'Lasso':
            from sklearn.linear_model import Lasso
            model_map['primary_models']['Lasso'] = SKLearnModel(Lasso(alpha=100, random_state=1), 'regression')
        elif model_name == 'Ridge':
            from sklearn.linear_model import Ridge
            model_map['primary_models']['Ridge'] = SKLearnModel(Ridge(alpha=0.1), 'regression')
        elif model_name == 'BayesianRidge':
            from sklearn.linear_model import BayesianRidge
            model_map['primary_models']['BayesianRidge'] = SKLearnModel(BayesianRidge(), 'regression')
        elif model_name == 'KNN':
            from sklearnex.neighbors import KNeighborsRegressor
            model_map['primary_models']['KNN'] = SKLearnModel(KNeighborsRegressor(n_neighbors=25), 'regression')
        elif model_name == 'AB':
            from sklearn.ensemble import AdaBoostRegressor
            model_map['primary_models']['AB'] = SKLearnModel(AdaBoostRegressor(random_state=1), 'regression')
        elif model_name == 'MLP':
            from sklearn.neural_network import MLPRegressor
            model_map['primary_models']['MLP'] = SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000), 'regression')
        elif model_name == 'RFR':
            # from sklearn.ensemble import RandomForestRegressor
            model_map['primary_models']['RFR'] = SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1), 'regression')
        elif model_name == 'SVR':
            from sklearnex.svm import SVR
            model_map['primary_models']['SVR'] = SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1), 'regression')
        elif model_name == 'StaticNaive':
            from models.naive import StaticNaiveModel
            model_map['primary_models']['StaticNaive'] = StaticNaiveModel()
        elif model_name == 'DNN':
            from models.neural import LightningNeuralNetModel
            from models.pytorch.neural_nets import MultiLayerPerceptron
            import torch.nn.functional as F
            model_map['primary_models']['DNN'] = LightningNeuralNetModel(
                MultiLayerPerceptron(
                    hidden_layers_ratio = [1.0], 
                    probabilities = False, 
                    loss_function = F.mse_loss), 
                max_epochs=15
            )
        
        elif model_name == 'LogisticRegression_two_class':
            from sklearn.linear_model import LogisticRegression
            model_map['primary_models']['LogisticRegression_two_class'] = SKLearnModel(LogisticRegression(C=10, random_state=1, solver='liblinear', max_iter=1000), 'classification')
        elif model_name == 'LogisticRegression_three_class':
            from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX
            model_map['primary_models']['LogisticRegression_three_class'] = SKLearnModel(LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1), 'classification')
        elif model_name == 'LDA':
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model_map['primary_models']['LDA'] = SKLearnModel(LinearDiscriminantAnalysis(), 'classification')
        elif model_name == 'KNN':
            from sklearn.neighbors import KNeighborsClassifier
            model_map['primary_models']['KNN'] = SKLearnModel(KNeighborsClassifier(), 'classification')
        elif model_name == 'CART':
            from sklearn.tree import DecisionTreeClassifier
            model_map['primary_models']['CART'] = SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1), 'classification')
        elif model_name == 'NB':
            from sklearn.naive_bayes import GaussianNB
            model_map['primary_models']['NB'] = SKLearnModel(GaussianNB(), 'classification')
        elif model_name == 'AB':
            from sklearn.ensemble import AdaBoostClassifier
            model_map['primary_models']['AB'] = SKLearnModel(AdaBoostClassifier(n_estimators=15), 'classification')
        elif model_name == 'RFC':
            model_map['primary_models']['RFC'] = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
        elif model_name == 'SVC':
            from sklearn.svm import SVC
            model_map['primary_models']['SVC'] = SKLearnModel(SVC(kernel='rbf', C=1e3, probability=True, random_state=1), 'classification')
        elif model_name == 'XGB_two_class':
            from xgboost import XGBClassifier
            from models.xgboost import XGBoostModel
            model_map['primary_models']['XGB_two_class'] = XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss'))
        elif model_name == 'LGBM':
            from lightgbm import LGBMClassifier
            model_map['primary_models']['LGBM'] = SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1), 'classification')
        elif model_name == 'StaticMom':
            from models.momentum import StaticMomentumModel
            model_map['primary_models']['StaticMom'] = StaticMomentumModel(allow_short=True)
        elif model_name == 'Average':
            from models.average import StaticAverageModel
            model_map['ensemble_models']['Average'] = StaticAverageModel()
    
    return model_map

