#%%
import pandas as pd
import pandas_ta as ta
from config.config import get_default_level_2_daily_config
from config.preprocess import preprocess_config
from data_loader.load_data import load_data

# %%
model_config, training_config, data_config = get_default_level_2_daily_config()
model_config, training_config, data_config = preprocess_config(model_config, training_config, data_config)

data_config['target_asset'] = data_config['assets'][0]
X, y, target_returns = load_data(**data_config)
# %%
X.ta.donchian()


# %%
X.ta.ema()
# %%
X.ta.adjusted = "ADA_USD_returns"

# %%
X.ta.sma(length=10)

# %%
X
# %%
X.ta.categories

# %%
ind_list = X.ta.indicators(as_list=True)

# %%
ind_list
# %%
X.ta.ao('ADA_USD_returns', length=10)
# %%
ta.ao()