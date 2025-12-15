import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Dense, GlobalMaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping



class Lstmcnnmodel:

    def __init__(self, n_features):

        #Plot learning curvec and if overfitt add regularizaetion and dropout layers
        #make a learning rate expo decay
        #Add normalization layers
        inp = Input(shape=(None, n_features))  # variable timesteps

        x = LSTM(64, return_sequences=True, activation="tanh")(inp)
        x = Conv1D(32, kernel_size=3, activation="relu")(x)
        x = MaxPooling1D(pool_size=2)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.3)(x)
        x = Dense(50, activation="relu")(x)
        x = Dropout(0.3)(x)
        out = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs=inp, outputs=out)

        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )


    def train(self, train_seq, val_seq, epochs=10):
        es = EarlyStopping(patience=3, restore_best_weights=True)

        return self.model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=epochs,
            callbacks=[es],
        )

    def predict(self, X_test):
        return self.model.predict(X_test)

    def summary(self):
        return self.model.summary()

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def save(self, path):
        self.model.save(path)

    # define a method that logs different model parameters and metrics to a dictionnary
    def log_params_metrics(self):
        log_dict = {}
        log_dict['model_summary'] = []
        self.model.summary(print_fn=lambda x: log_dict['model_summary'].append(x))
        log_dict['model_summary'] = "\n".join(log_dict['model_summary'])
        log_dict['optimizer'] = self.model.optimizer.get_config()
        return log_dict
