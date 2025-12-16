ALL_DAYS = [11, # 0
            12, # 0
            13, # .2876
            14, # .2442
            21, # .5000
            22, # .2200
            23, # .1747
            24, # .1203
            31, # 0
            32, # no glucose for ECG times
            33, # no glucose for ECG times
            34, # no glucose for ECG times
            41, # .1799
            42, # 0
            43, # .1042
            44, # 0
            51, # 0
            52, # 0
            53, # 0
            54, # 0
            61, # 0
            62, # 0
            63, # 0
            64, # .0730
            71, # 0
            72, # 0
            73, # 0
            74, # 0
            81, # .1902
            82, # .0242
            83, # .1489
            91, # no glucose for ECG times
            92, # .9048
            93, # .0577
            94 # no glucose for ECG times
            ]

DEMO_DAYS = [ # 11% of HG
            83, # potentially starts with HG and enough variation
            64 # really imbalanced but has some HG events
             ]

# TODO: decide on rules for TEST
# different people
# one day per person
# overall unbalanced
TEST_DAYS = [12,23,31,41, # 6.8% of HG
             53,62,73,81]

INVALID_DAYS = [32,33,34,91,94]  # days with no glucose measures for ECG times

# 12% in training
TRAIN_DAYS = [day for day in ALL_DAYS if day not in TEST_DAYS and day not in DEMO_DAYS and day not in INVALID_DAYS]



HG_DAYS = [13, 14, 21, 22, 23, 24, 41, 43,
           64, 81, 82, 83, 92, 93]

ZERO_DAYS = [11, 12, 31, 42, 44,
             51, 52, 53, 54, 61, 62, 63, 71, 72, 73, 74]

# 35 total days, 5 of which are invalid
# 10 days for test and demo: 8 test + 2 demo
# 20 days for cross-validation and training



import os
import pandas as pd

mlpreproc_params = {
            "ecg_dir": os.getenv("ECG_PATH"),
            "glucose_src": "local",
            "n_splits": 5,
            "chunk_size": pd.Timedelta(minutes=60),
            "step_size": pd.Timedelta(minutes=10),
            "forecast_window": pd.Timedelta(minutes=90),
            "roll_window_size": pd.Timedelta(minutes=40),
            "roll_step_size": pd.Timedelta(minutes=2),
            "suffix": f"rolling",
            "agg_funcs": ["mean", "std", "min", "max", "median", "skew", "kurtosis"],
            "random_state": 17,
        }
