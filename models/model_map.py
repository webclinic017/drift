from sklearn.linear_model import LinearRegression, Lasso, BayesianRidge, LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearnex.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearnex.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearnex.ensemble import RandomForestClassifier
from models.base import SKLearnModel
from models.momentum import StaticMomentumModel
from models.average import StaticAverageModel
from models.naive import StaticNaiveModel


model_map = {
    "regression_models": dict(
        LR = SKLearnModel(LinearRegression(n_jobs=-1)),
        Lasso = SKLearnModel(Lasso(alpha=100, random_state=1)),
        Ridge = SKLearnModel(Ridge(alpha=0.1)),
        BayesianRidge = SKLearnModel(BayesianRidge()),
        KNN = SKLearnModel(KNeighborsRegressor(n_neighbors=25)),
        AB = SKLearnModel(AdaBoostRegressor(random_state=1)),
        MLP = SKLearnModel(MLPRegressor(hidden_layer_sizes=(100,20), max_iter=1000)),
        RF = SKLearnModel(RandomForestRegressor(n_jobs=-1, random_state=1)),
        SVR = SKLearnModel(SVR(kernel='rbf', C=1e3, gamma=0.1)),
        StaticNaive = StaticNaiveModel(),
    ),
    "classification_models": dict(
        LR= SKLearnModel(LogisticRegression(C=10, random_state=1, max_iter=1000)),
        LDA= SKLearnModel(LinearDiscriminantAnalysis()),
        KNN= SKLearnModel(KNeighborsClassifier()),
        CART= SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1)),
        NB= SKLearnModel(GaussianNB()),
        AB= SKLearnModel(AdaBoostClassifier(n_estimators=15)),
        RF= SKLearnModel(RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1)),
        StaticMom= StaticMomentumModel(allow_short=True),
    ),     
    "classification_ensemble_models": dict(
        Ensemble_CART = SKLearnModel(DecisionTreeClassifier()),
        Ensemble_Average = StaticAverageModel(),
    ),
    "regression_ensemble_models": dict(
        Ensemble_Ridge = SKLearnModel(Ridge(alpha=0.1)),
        Ensemble_Average = StaticAverageModel(),
    )
}

model_names_classification = list(model_map["classification_models"].keys())
model_names_regression = list(model_map["regression_models"].keys())


def map_model_name_to_function(model_config:dict, method:str) -> dict:
    for level in ['level_1_models', 'level_2_models']:
        model_category = method + '_models' if level=='level_1_models' else method + '_ensemble_models'
        model_config[level] = [(model_name, model_map[model_category][model_name]) for model_name in  model_config[level]]

    return model_config