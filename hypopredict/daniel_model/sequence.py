import numpy as np
import tensorflow as tf

class PadAfterBatchSequence(tf.keras.utils.Sequence):
    """
    This is a Keras Sequence that pads variable-length sequences in a batch
    to the length of the longest sequence in that batch.
    Parameters:
    - X_list: list of numpy arrays, each of shape (timesteps, features)
    - y: numpy array of labels
    - batch_size: int, size of each batch
    - shuffle: bool, whether to shuffle data at the end of each epoch
    - maxlen: int or None, maximum length to pad/truncate sequences to
    - pad_value: float, value to use for padding

    """
    def __init__(self, X_list, y, class_weight=None, batch_size=32, shuffle=True, maxlen=None, pad_value=0.0):
        self.X_list = X_list
        self.y = np.asarray(y)
        self.class_wight = class_weight
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.maxlen = maxlen
        self.pad_value = pad_value

        self.indices = np.arange(len(self.X_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):

        batch_ids = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]

        X_batch_list = [self.X_list[i] for i in batch_ids]
        y_batch = self.y[batch_ids]

        # infer features dimension
        n_features = X_batch_list[0].shape[1]

        # pick pad length = max length in batch (optionally capped)
        batch_maxlen = max(x.shape[0] for x in X_batch_list)
        if self.maxlen is not None:
            batch_maxlen = min(batch_maxlen, self.maxlen)

        X_padded = np.full((len(X_batch_list), batch_maxlen, n_features),
                           fill_value=self.pad_value, dtype=np.float32)

        for j, x in enumerate(X_batch_list):
            # truncate if needed
            x_trunc = x[:batch_maxlen]
            X_padded[j, :x_trunc.shape[0], :] = x_trunc

        if getattr(self, "class_weight", None) is not None:
            sample_weight_batch = make_sample_weights(y_batch, self.class_weight)
            return X_padded, y_batch, sample_weight_batch


        return X_padded, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def make_sample_weights(y, class_weight):
        return np.array([class_weight[int(label)] for label in y], dtype=np.float32)
