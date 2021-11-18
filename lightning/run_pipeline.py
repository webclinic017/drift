
#%%
from create_dataset import load_format_data, create_dataloaders
from train_predict import train_model, predict
from models.built_in_models import create_TemporalFusionTransformer
from models.custom_model import create_FullyConnectedModel
from options import training_options, model_options_tft, model_options_fcn, dataset_options_tft, dataset_options_fcn

import warnings
warnings.filterwarnings("ignore")

#%%
def run_pipeline(model_name, data_dir):
    data = load_format_data(data_dir)  
    
    dataset_options, model_options, _create_model = select_model(model_name)
    training_dataset, train_dataloader, val_dataloader = create_dataloaders(data, dataset_options)
    
    model = _create_model( training_dataset, model_options )
    trainer = train_model(model, train_dataloader, val_dataloader, training_options)
    predict(trainer, model, val_dataloader)
    
#%%

def select_model(model_name):
    if model_name == "FullyConnectedLayer":
        return dataset_options_fcn, model_options_fcn, create_FullyConnectedModel
    elif model_name == "TemporalFusionTransformer":
        return dataset_options_tft, model_options_tft, create_TemporalFusionTransformer
    else:
        assert False, "No such model exists."
    

#%%
run_pipeline("FullyConnectedLayer", '../data/')

# #%%
# if __name__ == '__main__':
#     run_pipeline("FullyConnectedLayer", '../data/')

 