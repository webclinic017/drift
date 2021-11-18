from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer

from lightning.models.custom_model import FullyConnectedModel

training_options = dict(
    max_epochs=100,
    gpus=0,
    gradient_clip_val=0.1,
    limit_train_batches=30
)

model_options_fcn = dict(
    hidden_size=64,
    n_hidden_layers=2,
)

model_options_tft = dict(
    hidden_size=64,
    n_hidden_layers=2,
)

dataset_options_tft = dict(
    time_idx= 'time_idx',
    target= 'returns',
    # weight="weight",
    group_ids=[ 'ticker' ],
    min_encoder_length = 36, # this is the look-back window, see https://github.com/jdb78/pytorch-forecasting/issues/448
    max_encoder_length = 36,
    min_prediction_length = 1,
    max_prediction_length = 1,
    static_categoricals=[  ],
    static_reals=[  ],
    time_varying_known_categoricals=[ 'day_month', 'day_week', 'month' ],
    time_varying_known_reals=[ 'vol_10', 'vol_20', 'vol_30', 'vol_60', 'mom_10', 'mom_20', 'mom_30', 'mom_60', 'mom_90'],
    time_varying_unknown_categoricals=[  ],
    time_varying_unknown_reals=[ 'returns' ],
    allow_missing_timesteps=True,
    target_normalizer=GroupNormalizer(
        groups=['ticker'], transformation="softplus"),
    batch_size=128
)

dataset_options_fcn = dict(
    time_idx= 'time_idx',
    target= 'returns',
    # weight="weight",
    group_ids=[ 'ticker' ],
    min_encoder_length = 36, # this is the look-back window, see https://github.com/jdb78/pytorch-forecasting/issues/448
    max_encoder_length = 36,
    min_prediction_length = 1,
    max_prediction_length = 1,
    static_categoricals=[  ],
    static_reals=[  ],
    time_varying_known_categoricals=[ ],
    time_varying_known_reals=[ ],
    time_varying_unknown_categoricals=[  ],
    time_varying_unknown_reals=[ 'returns' ],
    allow_missing_timesteps=True,
    target_normalizer=GroupNormalizer(
        groups=['ticker'], transformation="softplus"),
    batch_size=128
)
