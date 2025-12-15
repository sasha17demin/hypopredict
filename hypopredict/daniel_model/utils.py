from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



def pad_and_array_lstm(X,y):
        """
        Pads X chunks

        Converts padded chunks and y to an array in order to feed into the model
        """
        #padding the data:
        maxlen = max([len(seq) for seq in X])
        feature_cols = X[0].columns.tolist() #Extract feature columns name
        X_array = [chunk[feature_cols].values for chunk in X]
        X_padded = pad_sequences(X_array, maxlen=maxlen, padding='post', dtype='float32', value = -111)
        #X_padded_stack= np.stack(X_padded, axis=0)
        #converting to an array to feed into the model
        y_array = np.array(y)
        return X_padded, y_array
