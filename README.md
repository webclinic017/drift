# Financial time series prediction

And end-to-end pipeline to train predictive Machine Learning models on financial (non-stationary, regime changing) time series. Includes feature selection and meta labelling.

## Why?

Machine learning on financial time series requires a fundamentally different approach compared to standard ML domains. The (small amount of) data is non-stationary, where the patterns frequently change, and it's extremely important to not to leak out-of-sample data into the test set.

There are very few open-source end-to-end machine learning pipelines that can be effectively used to train and evaluate ML models on financial time series. Among them are: [qlib](https://github.com/microsoft/qlib), [AlphaPy](https://github.com/ScottfreeLLC/AlphaPy).

This repo is different to them in a couple of angles:
- Feature extraction and selection is an important, pre-built step in the pipeline. Training models on irrelevant data will worsen the performance.
- Training and evaluation is done in a [walk-forward manner](https://en.wikipedia.org/wiki/Walk_forward_optimization). We argue that that one or two train/test split is not adoquate to evaluate an ML model's performance in a non-stationary, regime changing environment. The walk-forward methodology enables us to evaluate the model's performance on almost the whole time series. Combinatorial purged k-fold cross-validation can still have a place *within* a walk-forward training method - you can shuffle the past in any way you prefer, but never use future data to train the model, if you want objective evaluation of its performance. 
- The walk-forward training/evaluation methodology enables "online" (ever-changing) models, that adapt to the market environment. You can specify how frequently would you like to re-train the models.
- By default this framework always train multiple models, and average their predictions. Improves performance and adds a lot of robustness that's much needed in this environment.
- Meta-labeling (training a model to evaluate a lower level model's prediction for each timestamp) is a built-in feature - it improves performance a lot, and there are no open source implementation available as far as we're aware.

This project is inspired partially by [Marcos Lopez de Prado's Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) and [The Alpha Scientist's blogposts](https://alphascientist.com/).


## Installation

Use the conda environment file attached!:)


## Pipeline components

- Feature extraction
- Dimensionality reduction
- Feature selection
- Training Level-1 models
- Training Level-2 (Meta-label) model
- Evaluation of models
- Cross-sectional portfolio construction [IN PROGRESS]

## Weaknesses

Now that we assured that you should avoid lookahead bias at all costs, for practical reasons (to make this problem tractable), these are the things that are not completely lookahead-bias free:
- Dimensionality reduction
- Feature selection
