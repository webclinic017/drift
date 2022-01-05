from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearnex.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearnex.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearnex.ensemble import RandomForestClassifier
from models.base import SKLearnModel, LightningNeuralNetModel
from models.momentum import StaticMomentumModel
from models.average import StaticAverageModel
from models.naive import StaticNaiveModel
from models.pytorch.neural_nets import MultiLayerPerceptron
from xgboost import XGBClassifier
import torch.nn.functional as F


model_map = {
    "regression_models": dict(
        LR= SKLearnModel(LinearRegression(n_jobs=-1)),
        Lasso= SKLearnModel(Lasso(alpha=100, random_state=1)),
        Ridge= SKLearnModel(Ridge(alpha=0.1)),
        BayesianRidge= SKLearnModel(BayesianRidge()),
        KNN= SKLearnModel(KNeighborsRegressor(n_neighbors=25)),
        AB= SKLearnModel(AdaBoostRegressor(random_state=1)),
        MLP= SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
        RF= SKLearnModel(RandomForestRegressor(n_jobs=-1, max_depth=20, random_state=1)),
        SVR= SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1)),
        StaticNaive= StaticNaiveModel(),
        DNN = LightningNeuralNetModel(
            MultiLayerPerceptron(
                hidden_layers_ratio = [1.0], 
                probabilities = False, 
                loss_function = F.mse_loss), 
            max_epochs=15
        ) 
    ),
    "classification_models": dict(
        LR= SKLearnModel(LogisticRegression(C=10, random_state=1, max_iter=1000, n_jobs=-1)),
        LDA= SKLearnModel(LinearDiscriminantAnalysis()),
        KNN= SKLearnModel(KNeighborsClassifier()),
        CART= SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1)),
        NB= SKLearnModel(GaussianNB()),
        AB= SKLearnModel(AdaBoostClassifier(n_estimators=15)),
        RF= SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1)),
        XGB= SKLearnModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, use_label_encoder=True, objective='multi:softprob', eval_metric='mlogloss')),
        StaticMom= StaticMomentumModel(allow_short=True),
        Ensemble_Average= StaticAverageModel(),
    ),
}

model_names_classification = list(model_map["classification_models"].keys())
model_names_regression = list(model_map["regression_models"].keys())

default_feature_selector_regression = model_map['regression_models']['RF']
default_feature_selector_classification = model_map['classification_models']['RF']

