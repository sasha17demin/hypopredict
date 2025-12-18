import numpy as np
import pandas as pd
import os
from tensorflow import keras
import yaml

import hypopredict.chunk_preproc as cp
from hypopredict.cv import CV_splitter
from hypopredict import chunker
from hypopredict import labeler
from hypopredict.params import TRAIN_DAYS, DEMO_DAYS
from hypopredict.daniel_model.lstmcnn import Lstmcnnmodel
from hypopredict.daniel_model.sequence import PadAfterBatchSequence
from hypopredict.daniel_model.utils import build_model


days_train = TRAIN_DAYS
demo_days = DEMO_DAYS
ECG_PATH = os.getenv("ECG_PATH")

#TRAIN
splitter = CV_splitter(n_splits = 5,
                       ecg_dir = ECG_PATH,
                       glucose_src='local',
                       random_state = 3)
splits = splitter.get_splits(days_train)

splits_train = splits[:4]
splits_val = splits[4:]

split_train_chunkified = chunker.chunkify(splits_train.ravel(),
                                     chunk_size=pd.Timedelta(minutes=15),
                                     step_size=pd.Timedelta(minutes=1),
                                     ecg_dir=ECG_PATH)

split_val_chunkified = chunker.chunkify(splits_val.ravel(),
                                     chunk_size=pd.Timedelta(minutes=15),
                                     step_size=pd.Timedelta(minutes=1),
                                     ecg_dir=ECG_PATH)

split_train_labels = labeler.label_split(split_train_chunkified,
                                        glucose_src='gdrive',
                                        forecast_window=pd.Timedelta(minutes=60))

split_val_labels = labeler.label_split(split_val_chunkified,
                                        glucose_src='gdrive',
                                        forecast_window=pd.Timedelta(minutes=60))

X_train, y_train = cp.filter_and_stack(
                            split_train_chunkified, split_train_labels
                        )
X_val, y_val = cp.filter_and_stack(
                            split_val_chunkified, split_val_labels
                        )

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

class_weight = {
    0: 1.0,
    1: neg / pos
}

BATCH_SIZE = 16

train_seq = PadAfterBatchSequence(X_train, y_train, class_weight=class_weight, batch_size=BATCH_SIZE, shuffle=True, maxlen=BATCH_SIZE, pad_value = -111)
val_seq   = PadAfterBatchSequence(X_val, y_val, batch_size=BATCH_SIZE, shuffle=True, maxlen=BATCH_SIZE, pad_value = -111)

#Get the right .yaml file
with open("/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/config/baseline_lstmcnn.yaml", "r") as f:
    config = yaml.safe_load(f)

model = build_model(config, X_train[0].shape[1])

history = model.train(train_seq, val_seq, epochs=30)

probs_val = model.predict(val_seq).ravel()
probs_train = model.predict(train_seq).ravel()

#If you want to the model
#model.save("/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/checkpoints/baseline/lstmcnn_baseline_FW90.keras")

# DEMO
demo_day_64_chunkified = chunker.chunkify([64],
                                     chunk_size=pd.Timedelta(minutes=15),
                                     step_size=pd.Timedelta(minutes=1),
                                     ecg_dir=ECG_PATH)

demo_day_83_chunkified = chunker.chunkify([83],
                                     chunk_size=pd.Timedelta(minutes=15),
                                     step_size=pd.Timedelta(minutes=1),
                                     ecg_dir=ECG_PATH)

X_demo_day_64 = demo_day_64_chunkified[64]
X_demo_day_83 = demo_day_83_chunkified[83]

BATCH_SIZE = 16

demo_day_64_seq = PadAfterBatchSequence(X_demo_day_64, np.ones(len(X_demo_day_64)),
                                        batch_size=BATCH_SIZE, shuffle=False, maxlen=BATCH_SIZE, pad_value = -111)
demo_day_83_seq = PadAfterBatchSequence(X_demo_day_83, np.ones(len(X_demo_day_83)),
                                        batch_size=BATCH_SIZE, shuffle=False, maxlen=BATCH_SIZE, pad_value = -111)
# If you want to load a model
# model = keras.models.load_model("/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/checkpoints/baseline/lstmcnn_baseline.keras",
#                                 custom_objects= {"loss": Lstmcnnmodel.focal_loss(alpha=0.75, gamma=2)})

predict_64 = model.predict(demo_day_64_seq).ravel()
predict_83 = model.predict(demo_day_83_seq).ravel()

#If you want to save the results
# np.save("/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/results/predict_83.npy", predict_83)
# np.save("/home/danielfarkas/code/sasha17demin/hypopredict/hypopredict/daniel_model/results/predict_64.npy", predict_64)
