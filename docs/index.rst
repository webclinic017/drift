.. drift documentation master file, created by
   Daniel Szemerey and Mark Szulyovszky on Wed Feb 02 14:02:22 2022.



Welcome to drift's documentation!
==================================

TLDR: *Drift helps you train and make predictions on time-series data.*

.. figure:: images/pipeline.png
   :alt: Figure of Data transformation pipeline

   Pipeline of the entire process.

Drift is an **end-to-end***, **composable*** **modelling pipeline for financial time-series prediction**. It was designed for quantitative financial predictions.

Drift was specifically engineered not to fool the user: it uses walk-forward method for analysis and makes sure no future information is introduced.

Drift makes it easy to use state-of-the-art methods financial ML techniques, like the triple-barrier labeling method, ensembling models and adding bet-sizing into the mix (with meta-labeling). 

Drift has two level of usage: You can simply use existing models and transformations and just provide the data and the target asset, or you can customize your own pipeline.





Technical Explanation
==================================

Why Drift
--------------------------------

Machine learning on financial time series requires a fundamentally different approach compared to standard ML domains. 
The (small amount of) data is non-stationary, extremely noisy, where the patterns frequently change, and it's extremely important to not to leak out-of-sample data into the test set.

How Drift is different
--------------------------------

There are very few open-source end-to-end machine learning pipelines that can be effectively used to train and evaluate ML models on financial time series. Among them are: [qlib](https://github.com/microsoft/qlib), [AlphaPy](https://github.com/ScottfreeLLC/AlphaPy).

Drift is different to them in a couple of angles:

- All pre-processing steps are *online (up until a certain point),* ****so they never inject lookahead bias into the mix. (this is a major issue with finML papers)
- Feature extraction and selection are an important, pre-built step in the pipeline. Garbage in, garbage out!
- Evaluation is done in a [walk-forward manner](https://en.wikipedia.org/wiki/Walk_forward_optimization). We argue that that a train/validation/test split is not adequate to evaluate an ML model's performance in a non-stationary, regime changing environment. The walk-forward methodology enables us to evaluate the model's performance on almost the whole time series.
- Training can be done in any way possible, including Combinatorial purged k-fold cross-validation. You can shuffle the past in any way you prefer, but you can never use data from the future to train the model.
- Instead of training one model, you train tons of models **over time**, that are making predictions until they become obsolete. The walk-forward training/evaluation methodology enables "online" (ever-changing) models, that adapt to the market environment. You can specify how frequently would you like to re-train the models.
- Ensemble-by-default: train multiple models, and average their predictions. Improves performance and adds a lot of robustness in a low signal-to-noise environment, like financial time series.
- Bet sizing and Meta-labeling (training a model to evaluate a lower level model's prediction for each timestamp) is a built-in feature.

This project is inspired partially by [Marcos Lopez de Prado's Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) and [The Alpha Scientist's blogposts](https://alphascientist.com/).


External Links
--------------------------------

For more information refer
`here<www.python.org>`


.. py:function:: square(x)

   return the square of a function





Contents
==================================

.. toctree::
   :maxdepth: 2

   setup/index
   basic-usage/index
   advanced-usage/index

