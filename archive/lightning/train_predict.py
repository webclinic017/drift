import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import torch
from torch import nn


import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, '..')

from utils.load_data import load_files




def train_model(model, train_dataloader, val_dataloader, kwargs):
    print()
    print("Creating Trainer ===>", end="  ") 
    # define trainer with early stopping
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()
    
    trainer = pl.Trainer(
        **kwargs,
        callbacks=[lr_logger, early_stop_callback],
    )
    print("<===  Trainer Created")
    print("Finding Optimal LR  ===>", end="  ")
    # find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
    res = trainer.tuner.lr_find(
        model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
    )

    print(f"<=== suggested learning rate: {res.suggestion()}")
    fig = res.plot(show=True, suggest=True)
    fig.show()

    print("Training the model  ===>", end="  ")
    # fit the model
    trainer.fit(
        model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
    )
    
    print("<=== Training Finished")
    return trainer
    
    
def predict(trainer, model, val_dataloader):
    pass
    # best_model_path = trainer.checkpoint_callback.best_model_path
    # best_model = model.load_from_checkpoint(best_model_path)
    
    # # calcualte mean absolute error on validation set
    # actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    # predictions = best_model.predict(val_dataloader)
    # (actuals - predictions).abs().mean()
    
    # raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)
    # for idx in range(10):  # plot 10 examples
    #     best_model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=True)
    
    # predictions, x = best_model.predict(val_dataloader, return_x=True)
    # predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(x, predictions)
    # best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)
    