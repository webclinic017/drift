from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, Ridge
from sklearn.linear_model import LogisticRegression
from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX
from sklearn.tree import DecisionTreeClassifier
from sklearnex.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearnex.svm import SVR, SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearnex.ensemble import RandomForestClassifier
from models.sklearn import SKLearnModel
from models.neural import LightningNeuralNetModel
from models.momentum import StaticMomentumModel
from models.average import StaticAverageModel
from models.naive import StaticNaiveModel
from models.pytorch.neural_nets import MultiLayerPerceptron
from models.xgboost import XGBoostModel
from models.statsmodels import StatsModel
from xgboost import XGBClassifier
import torch.nn.functional as F
from lightgbm import LGBMClassifier
from statsmodels.tsa.api import ExponentialSmoothing



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
        LR_two_class= SKLearnModel(LogisticRegression(C=10, random_state=1, solver='liblinear', max_iter=1000)),
        LR_three_class= SKLearnModel(LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1)),
        LDA= SKLearnModel(LinearDiscriminantAnalysis()),
        KNN= SKLearnModel(KNeighborsClassifier()),
        CART= SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1)),
        NB= SKLearnModel(GaussianNB()),
        MNB = SKLearnModel(MultinomialNB()),
        AB= SKLearnModel(AdaBoostClassifier(n_estimators=15)),
        RF= SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1)),
        SVC = SKLearnModel(SVC(kernel='rbf', C=1e3, probability=True, random_state=1)),
        XGB_two_class= XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss')),
        LGBM = SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1)),
        StaticMom= StaticMomentumModel(allow_short=True),
        # ExpSmoothing = SKLearnModel(ExponentialSmoothing(trend='add', seasonal='add', seasonal_periods=30)),
    ),
    "ensemble_models": dict(
        Average= StaticAverageModel(),
    )
}

model_names_classification = list(model_map["classification_models"].keys())
model_names_regression = list(model_map["regression_models"].keys())

default_feature_selector_regression = model_map['regression_models']['RF']
default_feature_selector_classification = model_map['classification_models']['RF']

