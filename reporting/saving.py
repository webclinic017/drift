import pickle
import datetime
from config.types import Config
from typing import Optional
import os
import warnings
from training.types import PipelineOutcome


def save_models(pipeline_outcome: PipelineOutcome, config: Config) -> None:
    dict_for_pickle = dict()
    dict_for_pickle["config"] = config
    dict_for_pickle["pipeline_outcome"] = pipeline_outcome

    date_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

    if not os.path.exists("output/models"):
        warnings.warn("No folder exists, creating one.")
        os.makedirs("output/models")

    pickle.dump(dict_for_pickle, open("output/models/{}.p".format(date_string), "wb"))


def load_models(file_name: Optional[str]) -> tuple[PipelineOutcome, Config]:

    if file_name is None:
        warnings.warn(
            "No file name provided, will load latest models and configurations."
        )
        files_in_directory: list = os.listdir("output/models")

        assert len(files_in_directory) > 0, "No models found in output/models."
        file_name = sorted(files_in_directory)[-1]

    packacked_dict = pickle.load(open("output/models/{}".format(file_name), "rb"))

    config = packacked_dict.pop("config", None)
    pipeline_outcome = packacked_dict.pop("pipeline_outcome", None)

    return pipeline_outcome, config
