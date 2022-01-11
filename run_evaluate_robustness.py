from run_pipeline import run_pipeline
from config.config import get_default_ensemble_config, get_dev_config
import pandas as pd

all_results = []
all_predictions = []
for index in range(6):
    _, _, _, results_1, predictions_1, _ = run_pipeline(project_name='price-prediction', with_wandb = False, sweep = False, get_config= get_default_ensemble_config)
    all_results.append(results_1)
    all_predictions.append(predictions_1)

correlations = pd.Series()

for asset_name in [c for c in all_predictions[0].columns if 'ensemble' in c]:

    predictions_for_asset = pd.concat([preds[asset_name] for preds in all_predictions], axis=1)
    correlations[asset_name] = predictions_for_asset.corr().mean()[0]
    print("Correlation for asset ", asset_name, ": ", correlations[asset_name])

correlations["Overall"] = correlations.mean()
print("Average correlation across all assests: ", correlations.mean())

correlations.to_csv("output/correlations.csv")
