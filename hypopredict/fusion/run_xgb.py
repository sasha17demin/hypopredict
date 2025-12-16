import numpy as np
import pandas as pd
import os

from hypopredict.cv import CV_splitter
from hypopredict import chunker
from hypopredict import labeler
from hypopredict.params import TRAIN_DAYS

from hypopredict.cv import CrossValidator

ECG_PATH = os.getenv('ECG_PATH')

#######
# chunking strategy
CHUNK_SIZE = pd.Timedelta(minutes=60)
STEP_SIZE = pd.Timedelta(minutes=10)

#######
# labeling strategy
FORECAST_WINDOW = pd.Timedelta(minutes=90)


######
# rolling features
WINDOW_SIZE_FEATURES = pd.Timedelta(minutes=40)
STEP_SIZE_FEATURES = pd.Timedelta(minutes=2)

##### TODO: class ML_model_pipeline():
# initialize CV splitter
splitter = CV_splitter(n_splits = 5,
                       ecg_dir = ECG_PATH,
                       glucose_src='local',
                       random_state = 17)
# get splits
splits = splitter.get_splits(TRAIN_DAYS)

crossval = CrossValidator(splits = splits)

splits_prepped = crossval.chunkify_label_stack(
    chunk_size=CHUNK_SIZE,
    step_size=STEP_SIZE,
    ecg_dir=ECG_PATH,
    glucose_src='local',
    forecast_window=FORECAST_WINDOW,
    roll_window_size=WINDOW_SIZE_FEATURES,
    roll_step_size=STEP_SIZE_FEATURES,
    suffix=f'roll{WINDOW_SIZE_FEATURES.components.minutes}min',
    agg_funcs=['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']
)

# initialize XGBoost model
from xgboost import XGBClassifier
xgb = XGBClassifier(
    n_estimators=777,
    max_depth=5,
    reg_lambda=0.1,
    learning_rate=0.2,
    eval_metric='logloss',
    random_state=17,
    verbose=False
)

cv_results_xgb = crossval.validate_model_cv(xgb, splits_prepped,
                                        resample=True,
                                        desired_pos_ratio=0.4,
                                        reduction_factor=0.7)

print("XGBoost mean CV PR AUCs:")
print(np.mean(cv_results_xgb['val_pr_aucs']))
print("XGBoost mean CV ave Prec:")
print(np.mean(cv_results_xgb['val_ave_precisions']))
print("Baseline:")
print(np.mean(crossval._get_split_mean_labels(splits_prepped)))
