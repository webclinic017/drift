from sklearn.model_selection import train_test_split

from config.types import Config, RawConfig
from config.preprocess import preprocess_config
from config.presets import get_default_config, get_minimal_config

from data_loader.load import load_data
from data_loader.process import check_data

from labeling.process import label_data

from sklearn.metrics import accuracy_score


def run_sklearn_pipeline(raw_config: RawConfig):
    config = preprocess_config(raw_config)
    run_sklearn_training(config)


def run_sklearn_training(config: Config):

    print("---> Load data, check for validity")
    X, returns = load_data(
        assets=config.assets,
        other_assets=config.other_assets,
        exogenous_data=config.exogenous_data,
        target_asset=config.target_asset,
        load_non_target_asset=config.load_non_target_asset,
        own_features=config.own_features,
        other_features=config.other_features,
        exogenous_features=config.exogenous_features,
        start_date=config.start_date,
    )

    assert check_data(X, config) == True, "Data is not valid."

    print("---> Filter for significant events when we want to trade, and label data")
    events, X, y, forward_returns = label_data(
        event_filter=config.event_filter,
        event_labeller=config.labeling,
        X=X,
        returns=returns,
        remove_overlapping_events=config.remove_overlapping_events,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    print("---> Train directional models")

    for transformation in config.transformations:
        X_train = transformation.fit_transform(X_train, y_train)

    for transformation in config.transformations:
        X_test = transformation.transform(X_test)

    config.directional_model.fit(X_train, y_train)
    preds = config.directional_model.predict(X_test)
    print(accuracy_score(y_test, preds))


if __name__ == "__main__":
    run_sklearn_pipeline(
        raw_config=get_default_config(),
    )
