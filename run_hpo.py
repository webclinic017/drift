from config.types import RawConfig
from run_pipeline import run_pipeline
from config.presets import get_default_config
import optuna
from utils.helpers import powerset

config = get_default_config()
config.save_models = False


def train(trial):
    search_space = {
        "n_features_to_select": trial.suggest_int("n_features_to_select", 10, 200),
        "forecasting_horizon": trial.suggest_int("forecasting_horizon", 3, 300),
        "own_features": trial.suggest_categorical(
            "own_features", powerset(["z_score", "level_2", "level_1"])
        ),
        "other_features": trial.suggest_categorical(
            "other_features", powerset(["z_score", "level_2", "level_1"])
        ),
        "labeling": trial.suggest_categorical(
            "labeling", ["two_class", "three_class_balanced", "three_class_imbalanced"]
        ),
        "scaler": trial.suggest_categorical(
            "scaler",
            ["normalize", "minmax", "standardize", "robust", "box-cox", "quantile"],
        ),
    }
    trial_config = RawConfig(**vars(config) | search_space)
    outcome, _ = run_pipeline(
        project_name="test", with_wandb=False, raw_config=trial_config
    )
    return 1 - outcome.get_output_stats()["sharpe"]


study = optuna.create_study()
study.optimize(train, n_trials=40)
print(study.best_params)
study.trials_dataframe().to_csv("output/trials.csv")
