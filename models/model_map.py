from models.sklearn import SKLearnModel
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.ensemble import RandomForestRegressor
from .base import Model

default_feature_selector_classification = SKLearnModel(
    RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1)
)


def get_model(model_name: str) -> Model:
    def set_name(model: Model) -> Model:
        model.name = model_name
        return model

    if model_name == "LogisticRegression_two_class":
        from sklearn.linear_model import LogisticRegression

        return set_name(
            SKLearnModel(
                LogisticRegression(
                    C=10, random_state=1, solver="liblinear", max_iter=1000
                )
            )
        )
    elif model_name == "LogisticRegression_three_class":
        from sklearnex.linear_model import LogisticRegression as LogisticRegression_EX

        return set_name(
            SKLearnModel(
                LogisticRegression_EX(C=10, random_state=1, max_iter=1000, n_jobs=-1)
            )
        )
    elif model_name == "LDA":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return set_name(SKLearnModel(LinearDiscriminantAnalysis()))
    elif model_name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier

        return set_name(SKLearnModel(KNeighborsClassifier()))
    elif model_name == "CART":
        from sklearn.tree import DecisionTreeClassifier

        return set_name(
            SKLearnModel(DecisionTreeClassifier(max_depth=15, random_state=1))
        )
    elif model_name == "NB":
        from sklearn.naive_bayes import GaussianNB

        return set_name(SKLearnModel(GaussianNB()))
    elif model_name == "AB":
        from sklearn.ensemble import AdaBoostClassifier

        return set_name(SKLearnModel(AdaBoostClassifier(n_estimators=15)))
    elif model_name == "RFC":
        return set_name(
            SKLearnModel(
                RandomForestClassifier(n_jobs=-1, max_depth=20, random_state=1)
            )
        )
    elif model_name == "SVC":
        from sklearn.svm import SVC

        return set_name(
            SKLearnModel(SVC(kernel="rbf", C=1e3, probability=True, random_state=1))
        )
    # elif model_name == 'XGB_two_class':
    #     from xgboost import XGBClassifier
    #     from models.xgboost import XGBoostModel
    #     return set_name(XGBoostModel(XGBClassifier(n_jobs=-1, max_depth = 20, random_state=1, objective='binary:logistic', use_label_encoder= False, eval_metric='mlogloss')))
    elif model_name == "LGBM":
        from lightgbm import LGBMClassifier

        return set_name(
            SKLearnModel(LGBMClassifier(n_jobs=-1, max_depth=20, random_state=1))
        )

    elif model_name == "AutoML":
        from supervised.automl import AutoML

        return set_name(
            SKLearnModel(
                AutoML(
                    total_time_limit=60,
                    mode="Compete",
                    algorithms=[
                        "Baseline",
                        "Linear",
                        "Random Forest",
                        "Extra Trees",
                        "LightGBM",
                        "CatBoost",
                        "Neural Network",
                        "Nearest Neighbors",
                    ],
                    validation_strategy={
                        "validation_type": "split",
                        "train_ratio": 0.75,
                        "shuffle": False,
                        "stratify": True
                    },
                    eval_metric="f1",
                )
            )
        )
    elif model_name == "StaticMom":
        from models.momentum import StaticMomentumModel

        return set_name(StaticMomentumModel(allow_short=True))
    else:
        raise Exception(f"Model {model_name} not found")
