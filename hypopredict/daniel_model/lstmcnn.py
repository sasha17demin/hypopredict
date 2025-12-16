import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Dropout, Dense, GlobalMaxPooling1D, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping



class Lstmcnnmodel:

    @staticmethod
    def focal_loss(alpha=0.75, gamma=2.0):
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            eps = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)

            pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
            w  = tf.where(tf.equal(y_true, 1), alpha, 1 - alpha)

            return -tf.reduce_mean(w * tf.pow(1 - pt, gamma) * tf.math.log(pt))
        return loss


    def __init__(self, config, n_features):

        #Plot learning curvec and if overfitt add regularizaetion and dropout layers
        #make a learning rate expo decay
        #Add normalization layers
        self.config = config

        arch = config['architecture']
        train_cfg = config['training']

        # Keras.Tuner similar approach than gridsearch but for DL
        # Optuna more powerful but more complex

        inp = Input(shape=(None, n_features))  # variable timesteps

        x = LSTM(arch['lstm_units'], return_sequences=True, activation="tanh")(inp)
        x = Conv1D(arch['conv_filters'], kernel_size=arch['kernel_size'], activation="relu")(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=arch['pool_size'])(x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(arch['dropout'])(x)
        x = Dense(arch['dense_units'], activation="relu")(x)
        x = Dropout(arch['dropout'])(x)
        out = Dense(1, activation="sigmoid")(x)

        self.model = Model(inputs=inp, outputs=out)

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=float(train_cfg["learning_rate"])),
            loss = Lstmcnnmodel.focal_loss(train_cfg.get("focal_alpha", 0.25), train_cfg.get("focal_gamma", 2.0)),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc", curve="PR"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
            ],
        )


    def train(self, train_seq, val_seq, epochs=10):
        es = EarlyStopping( monitor="val_pr_auc",mode="max",patience=8, restore_best_weights=True)

        return self.model.fit(
            train_seq,
            validation_data=val_seq,
            epochs=epochs,
            callbacks=[es]
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
