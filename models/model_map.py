from models.sklearn import SKLearnModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

model_map = {
    "regression_models": dict(),
    "classification_models": dict(),
    "ensemble_models": dict()
}

def get_model_map(config:dict):
    
    if len(config['primary_models']) > 0 and isinstance(config['primary_models'][0], str):
        print("Going to Load models")
        combined_list = config['primary_models'] + config['meta_labeling_models'] + [config['ensemble_model']]
        for model_name in combined_list:
            if model_name == 'LR':
                from sklearn.linear_model import LinearRegression
                model_map['regression_models']['LR'] = SKLearnModel(LinearRegression(n_jobs=-1))
            elif model_name == 'Lasso':
                from sklearn.linear_model import Lasso
                model_map['regression_models']['Lasso'] = SKLearnModel(Lasso(alpha=100, random_state=1))
            elif model_name == 'Ridge':
                from sklearn.linear_model import Ridge
                model_map['regression_models']['Ridge'] = SKLearnModel(Ridge(alpha=0.1))
            elif model_name == 'BayesianRidge':
                from sklearn.linear_model import BayesianRidge
                model_map['regression_models']['BayesianRidge'] = SKLearnModel(BayesianRidge())
            elif model_name == 'KNN':
                from sklearnex.neighbors import KNeighborsRegressor
                model_map['regression_models']['KNN'] = SKLearnModel(KNeighborsRegressor(n_neighbors=25))
            elif model_name == 'AB':
                from sklearn.ensemble import AdaBoostRegressor
                model_map['regression_models']['AB'] = SKLearnModel(AdaBoostRegressor(random_state=1))
            elif model_name == 'MLP':
                from sklearn.neural_network import MLPRegressor
                model_map['regression_models']['MLP'] = SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000))
            elif model_name == 'RFR':
                # from sklearn.ensemble import RandomForestRegressor
                model_map['regression_models']['RFR'] = SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1))
            elif model_name == 'SVR':
                from sklearnex.svm import SVR
                model_map['regression_models']['SVR'] = SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1))
            elif model_name == 'StaticNaive':
                from models.naive import StaticNaiveModel
                model_map['regression_models']['StaticNaive'] = StaticNaiveModel()
            elif model_name == 'DNN':
                from models.neural import LightningNeuralNetModel
                from models.pytorch.neural_nets import MultiLayerPerceptron
                import torch.nn.functional as F
                model_map['regression_models']['DNN'] = LightningNeuralNetModel(
                    MultiLayerPerceptron(
                        hidden_layers_ratio = [1.0], 
                        probabilities = False, 
                        loss_function = F.mse_loss), 
                    max_epochs=15
                )
            
            
            elif model_name == 'LR_two_class':
                from sklearn.linear_model import LogisticRegression
                model_map['classification_models']['LR_two_class'] = SKLearnModel(LogisticRegression(C=10, random_state=1, solver='liblinear', max_iter=1000))
            elif model_name == 'LR_three_class':
                from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX
                model_map['classification_models']['LR_three_class'] = SKLearnModel(LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1))
            elif model_name == 'LDA':
                from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                model_map['classification_models']['LDA'] = SKLearnModel(LinearDiscriminantAnalysis())
            elif model_name == 'KNN':
                from sklearn.neighbors import KNeighborsClassifier
                model_map['classification_models']['KNN'] = SKLearnModel(KNeighborsClassifier())
            elif model_name == 'CART':
                from sklearn.tree import DecisionTreeClassifier
                model_map['classification_models']['CART'] = SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1))
            elif model_name == 'NB':
                from sklearn.naive_bayes import GaussianNB
                model_map['classification_models']['NB'] = SKLearnModel(GaussianNB())
            elif model_name == 'AB':
                from sklearn.ensemble import AdaBoostClassifier
                model_map['classification_models']['AB'] = SKLearnModel(AdaBoostClassifier(n_estimators=15))
            elif model_name == 'RFC':
                # from sklearn.ensemble import RandomForestClassifier
                model_map['classification_models']['RFC'] = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1))
            elif model_name == 'SVC':
                from sklearn.svm import SVC
                model_map['classification_models']['SVC'] = SKLearnModel(SVC(kernel='rbf', C=1e3, probability=True, random_state=1))
            elif model_name == 'XGB_two_class':
                from xgboost import XGBClassifier
                from models.xgboost import XGBoostModel
                model_map['classification_models']['XGB_two_class'] = XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss'))
            elif model_name == 'LGBM':
                from lightgbm import LGBMClassifier
                model_map['classification_models']['LGBM'] = SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1))
            elif model_name == 'StaticMom':
                from models.momentum import StaticMomentumModel
                model_map['classification_models']['StaticMom'] = StaticMomentumModel(allow_short=True)
            elif model_name == 'Average':
                from models.average import StaticAverageModel
                model_map['ensemble_models']['Average'] = StaticAverageModel()
           
    
    model_names_classification = list(model_map["classification_models"].keys())
    model_names_regression = list(model_map["regression_models"].keys())

    
    default_feature_selector_classification = SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1))
    default_feature_selector_regression = SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1))

    
    return model_map, model_names_classification, model_names_regression, default_feature_selector_regression, default_feature_selector_classification

