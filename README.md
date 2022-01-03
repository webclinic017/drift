# Financial time series prediction

And end-to-end pipeline to train predictive Machine Learning models on financial (non-stationary, regime changing) time series.

## Why?

Machine learning on financial time series require a fundamentally different approach than used in other ML domains. The data is non-stationary, where the patterns frequently change, and it's extremely important to not to leak out-of-sample data into the training set objective evaluation is important. 

There are very few open-source end-to-end machine learning pipelines that can be effectively used to train and evaluate ML models on financial time series. Among them are[qlib](https://github.com/microsoft/qlib), [AlphaPy](https://github.com/ScottfreeLLC/AlphaPy).

This repo is different to them in a couple of angles:
- Feature extraction and selection is an important, pre-built step in the pipeline. Training models on 
- Training and evaluation is done in a [walk-forward manner](https://en.wikipedia.org/wiki/Walk_forward_optimization). We argue that that one or two train/test split is not adoquate to evaluate an ML model's performance in a non-stationary, regime changing environment. The walk-forward methodology enables us to evaluate the model's performance on almost the whole time series.
- The walk-forward training/evaluation methodology enables "online" (ever-changing) models, that adapt to the market environment. You can specify how frequently would you like to re-train the models.


This project is inspired partially by [Marcos Lopez de Prado's Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) and [The Alpha Scientist's blogposts](https://alphascientist.com/).



## Installation

Use the conda environment file attached!:)


## Pipeline components

- Feature extraction
- Dimensionality reduction
- Feature selection
- Training Level-1 models
- Training Level-2 (Ensemble) model
- Evaluation of models
- Cross-sectional portfolio construction [IN PROGRESS]