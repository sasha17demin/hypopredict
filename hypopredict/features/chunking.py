"""
Time-series chunking and sliding window operations.

Renamed from train_test_split.py for honest naming (it chunks data, doesn't split).
Handles sliding window segmentation of sensor data and label generation.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Union

from hypopredict.core.person import Person
from hypopredict.core.data_types import PersonDay


def chunkify(
    person_days: Union[List[int], List[PersonDay]],
    chunk_size: pd.Timedelta,
    step_size: pd.Timedelta,
    ecg_dir: str
) -> Dict[Union[int, PersonDay], List[pd.DataFrame]]:
    """
    Split sensor data into overlapping chunks for multiple person-days.
    
    Args:
        person_days: List of person-day identifiers (legacy int or PersonDay objects)
        chunk_size: Duration of each chunk
        step_size: Slide step between consecutive chunks
        ecg_dir: Directory containing ECG data files
        
    Returns:
        Dictionary mapping person_day to list of chunk DataFrames
    """
    chunks_all_days = {}
    for person_day in person_days:
        person_day, chunks_person_day = chunkify_day(
            person_day,
            chunk_size=chunk_size,
            step_size=step_size,
            ecg_dir=ecg_dir
        )
        chunks_all_days[person_day] = chunks_person_day

    return chunks_all_days


def chunkify_day(
    person_day: Union[int, PersonDay],
    chunk_size: pd.Timedelta,
    step_size: pd.Timedelta,
    ecg_dir: str
) -> Tuple[Union[int, PersonDay], List[pd.DataFrame]]:
    """
    Split sensor data for a single person-day into overlapping chunks.
    
    Args:
        person_day: Person-day identifier (legacy int or PersonDay object)
        chunk_size: Duration of each chunk
        step_size: Slide step between consecutive chunks
        ecg_dir: Directory containing ECG data files
        
    Returns:
        Tuple of (person_day, list of chunk DataFrames)
    """
    # Handle both legacy int and PersonDay types
    if isinstance(person_day, PersonDay):
        person_id = person_day.person_id
        day = person_day.day
    else:
        # Legacy integer format (e.g., 35 = person 3, day 5)
        person_id = person_day // 10
        day = person_day % 10

    person = Person(person_id, ecg_dir=ecg_dir)
    person.load_ECG_day(day=day)

    chunks_person_day = chunkify_df(
        person.ecg[day],
        chunk_size=chunk_size,
        step_size=step_size
    )

    return person_day, chunks_person_day


def chunkify_df(
    df: pd.DataFrame,
    chunk_size: pd.Timedelta,
    step_size: pd.Timedelta
) -> List[pd.DataFrame]:
    """
    Function that splits sensor df (ECG, Breathing, Acceleration)
    into overlapping chunks using a sliding window.

    Args:
        df: pd.DataFrame - dataframe with sensor data (datetime-indexed)
        chunk_size: pd.Timedelta - duration of each chunk
        step_size: pd.Timedelta - slide step between consecutive chunks
        
    Returns:
        list of pd.DataFrame - list of overlapping chunks
    """
    chunks = []
    
    # Calculate sampling rate from data (assumes uniform sampling)
    df_sampling_rate = 1 / (df.index[1] - df.index[0]).total_seconds()  # in Hz
    
    # Convert timedeltas to number of samples
    chunk_size_samples = int(chunk_size.total_seconds() * df_sampling_rate)
    step_size_samples = int(step_size.total_seconds() * df_sampling_rate)

    # Slide through dataframe with step size
    for start in range(0, df.shape[0], step_size_samples):
        end = start + chunk_size_samples

        # if the end index is within the dataframe, take the slice as chunk
        if end < df.shape[0]:
            chunk = df.iloc[start:end]
        # otherwise, take the slice from start to the end of the dataframe
        else:
            chunk = df.iloc[start:]

        chunks.append(chunk)

    return chunks


def get_HG_onset_times(glucose_df: pd.DataFrame, threshold: float = 3.9) -> List[pd.Timestamp]:
    """
    Function that gets the onset times of hypoglycemia events
    from the glucose dataframe.
    
    An onset is the first time glucose drops below threshold, with at least
    5 minutes separation from previous onset to count as a new event.

    Args:
        glucose_df: pd.DataFrame - dataframe with glucose data
                    Should have 'glucose' column and datetime index
        threshold: float - glucose threshold for hypoglycemia (mmol/L)

    Returns:
        list of pd.Timestamp - list of onset times
    """
    times = glucose_df[glucose_df['glucose'] <= threshold].index

    times_onset = []
    prev_time = None

    for time in times:
        # Check if this time is at least 5 minutes after the last recorded onset
        # that means it's a new onset
        if prev_time is None or (time - prev_time) > pd.Timedelta(minutes=5):
            times_onset.append(time)
        prev_time = time

    return times_onset


def generate_target_labels(
    chunks: List[pd.DataFrame],
    onset_times: List[pd.Timestamp],
    forecast_window: pd.Timedelta
) -> List[int]:
    """
    Function that generates target labels for each chunk based on onset times
    and forecast window. If an onset time falls within the forecast window
    after the end of a chunk, the label for that chunk is 1, otherwise 0.
    
    Prediction target: Will there be an HG event in the next forecast_window?

    Args:
        chunks: list of pd.DataFrame - list of overlapping chunks
                                        Should have index as datetime
        onset_times: list of pd.Timestamp - list of Hypoglycemia onset times
        forecast_window: pd.Timedelta - forecast window duration

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
    assert len(target_labels) == len(chunks)

    return target_labels


def train_test_split_chunks(
    chunks: List[pd.DataFrame],
    chunk_size: int,
    step_size: int,
    target_labels: List[int],
    forecast_window: pd.Timedelta,
    sampling_rate: int = 250,
    test_size: float = 0.2
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[int], List[int]]:
    """
    Function that splits chunks and target labels into train and test sets
    while respecting the forecast window constraint to avoid data leakage.
    
    Ensures train chunks don't leak into test period via the forecast window.

    Args:
        chunks: list of pd.DataFrame - list of overlapping chunks
        chunk_size: int - size of each chunk in samples
        step_size: int - step size between chunks in samples
        target_labels: list of int - list of target labels (0 or 1) for each chunk
        forecast_window: pd.Timedelta - forecast window duration
        sampling_rate: int - sampling rate of the sensor data in Hz
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
    # by at least forecast_window and chunk_size to avoid data leakage
    last_train_chunk_index = split_index - np.ceil(
        (forecast_window_freq + chunk_size) / step_size
    ).astype(int) - 1

    X_train = chunks[:last_train_chunk_index]
    y_train = target_labels[:last_train_chunk_index]

    return X_train, X_test, y_train, y_test
