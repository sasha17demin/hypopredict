import pandas as pd
import numpy as np


def hello():    return "Hello from hypopredict.train_test_split module!!"

# split sendsor data into overlapping chunks of CHUNK_SIZE every STEP_SIZE seconds


def chunkify(df: pd.DataFrame, chunk_size: int, step_size: int) -> list[pd.DataFrame]:
    """
    Function that splits sensor df (ECG, Breathing, Accelaration)
    into overlapping chunks to generate X_train.

    Args:
        df: pd.DataFrame - dataframe with sensor data
        chunk_size: int - size of each chunk in number of samples
        step_size: int - step size in number of samples
    Returns:
        list of pd.DataFrame - list of overlapping chunks
    """

    chunks = []

    # take the starting index from 0 to the end of the dataframe with step size slide
    for start in range(0, df.shape[0], step_size):
        end = start + chunk_size  # ending index of the chunk

        # if the end index is within the dataframe, take the slice as chunk
        if end < df.shape[0]:
            chunk = df.iloc[start:end]
        # otherwise, take the slice from start to the end of the dataframe
        else:
            chunk = df.iloc[start:]

        chunks.append(chunk)  # for each start index, append the chunk to the list

    return chunks







###### getting HG onset times ######



# TODO: HG event is (a) glucose < 3.9 and (b) lasts for at least 15 minutes
# currently only (a) is implemented
# TODO: unique event identifier across subjects
# to make sure we don't include the same event in train and test


def get_HG_onset_times(glucose_df: pd.DataFrame, threshold: float = 3.9) -> list:
    """
    Function that gets the onset times of hypoglycemia events
    from the glucose dataframe.

    Args:
        glucose_df: pd.DataFrame - dataframe with glucose data
                    Should have 'glucose' column and datetime index

        threshold: float - glucose threshold for hypoglycemia

    Returns:
        list of pd.Timestamp - list of onset times
    """

    times = glucose_df[glucose_df['glucose'] <= threshold].index
    #times = pd.to_datetime(times, format='%Y-%m-%d %H:%M:%S:%f')

    times_onset = []
    prev_time = None

    for time in times:
        # Check if this time is at least 5 minutes after the last recorded onset
        # that means it's a new onset
        if prev_time is None or (time - prev_time) > pd.Timedelta(minutes=5):
            times_onset.append(time)
        prev_time = time

    return times_onset







######### Generating target labels #########

# target label is 1 if NEXT chunk contains any of the onset times

#       1       0       0       0       0
#           |  ot   |       |       |       |
#           |       |       |       |       |
#     chunk1  chunk2  chunk3  chunk4  chunk5

# onset_time: first time glucose < 3.9


# TODO: ongoing event identifier

def generate_target_labels(chunks: list[pd.DataFrame],
                           onset_times: list[pd.Timestamp],
                           forecast_window: pd.Timedelta) -> list[int]:
    """
    Function that generates target labels for each chunk based on onset times
    and forecast window. If an onset time falls within the forecast window
    after the end of a chunk, the label for that chunk is 1, otherwise 0.

    Args:
        chunks: list of pd.DataFrame - list of overlapping chunks
                                        Should have index as datetime
        onset_times: list of pd.Timestamp - list of Hypoglycemia onset times
        forecast_window: pd.Timedelta - forecast window duration as timedelta

    Returns:
        list of int - list of target labels (0 or 1) for each chunk
    """

    target_labels = []

    for chunk in chunks:

        chunk_end_time = chunk.index[-1]
        label = 0

        # for each onset time, check if it falls within the forecast window
        # after the end of a given chunk
        for onset_time in onset_times:
            if chunk_end_time < onset_time < chunk_end_time + forecast_window:
                label = 1
                break

        target_labels.append(label)

    # check that the number of labels matches the number of chunks
    assert(len(target_labels) == len(chunks))

    return target_labels







######## Train-Test Split #########

# last date of the last train chunk is t1
# the first date of the first test chunk is t1+FORECAST_WINDOW

def train_test_split_chunks(chunks: list[pd.DataFrame],
                            chunk_size: int,
                            step_size: int,
                            target_labels: list[int],
                            forecast_window: pd.Timedelta,
                            sampling_rate: int = 250,
                            test_size: float = 0.2) -> tuple:
    """
    Function that splits chunks and target labels into train and test sets
    while respecting the forecast window constraint.

    Args:
        chunks: list of pd.DataFrame - list of overlapping chunks
        step_size: int - step size between chunks in samples
        sampling_rate: int - sampling rate of the sensor data in Hz

        target_labels: list of int - list of target labels (0 or 1) for each chunk
        forecast_window: pd.Timedelta - forecast window duration as timedelta
        test_size: float - proportion of the dataset to include in the test split

    Returns:
        tuple - (X_train, X_test, y_train, y_test)
    """

    total_chunks = len(chunks)

    # leave test_size proportion of chunks for testing
    num_test_chunks = int(np.ceil(total_chunks * test_size))

    # take all the last test_size% of chunks and labels as test set
    split_index = total_chunks - num_test_chunks

    X_test = chunks[split_index:]
    y_test = target_labels[split_index:]




    # make sure forecast window is in number of samples using step_size
    forecast_window_freq = int(forecast_window.total_seconds() * sampling_rate)

    # move the last index of the last train chunk to the left
    # by at least forecast_window and chunk_size
    last_train_chunk_index = split_index - np.ceil( (forecast_window_freq + chunk_size) / step_size).astype(int) - 1

    X_train = chunks[:last_train_chunk_index]
    y_train = target_labels[:last_train_chunk_index]


    return X_train, X_test, y_train, y_test
