"""Module for generating HG target labels
for sensor data chunks based on hypoglycemia events.
"""

import pandas as pd
import numpy as np

from hypopredict.core.person import Person

#### We know how to generate label for 1 chunk
### Refactor -> function: list[chunks] --> list[labels]


def label_day(day: int,
              chunks: list[pd.DataFrame],
                          # person, will instantiate from dict key
                          forecast_window: pd.Timedelta,
                          glucose_src) -> list[int]:
    """
    Function that generates target labels for each chunk based on hypoglycemia events
    in the forecast window after the end of the chunk.

    Args:
        split_chunkified: dict[int, list[pd.DataFrame]] - dictionary of overlapping chunks
                                        Should have index as datetime
        person: Person - Person object with hg_events_with_ECG attribute
        forecast_window: pd.Timedelta - forecast window duration as timedelta

    Returns:
        list of int - list of target labels (0 or 1) for each chunk
    """

    # instantiate Person from split_chunkified key
    person_id = day // 10  # first digit of key
      # avoid circular import
    person = Person(person_id, ecg_dir=None)  # ecg_dir not needed here
    person.load_HG_data(glucose_src=glucose_src)  # glucose_src not needed here

    target_labels = []

    for chunk in chunks:

        chunk_end_time = chunk_end(chunk)
        forecast_window_end = chunk_end_time + forecast_window

        # subset hg_events_with_ECG to forecast window
        hg_events_forecast = person.hg_events[chunk_end_time: forecast_window_end]

        # if no glucose data in forecast window, label is -111
        if hg_events_forecast.empty:
            target_labels.append(-111)
            continue

        # label is 1 if any hg event in forecast window, else 0
        chunk_label = int(hg_events_forecast['is_hg'].max())

        target_labels.append(chunk_label)

    return np.array(target_labels)





# helper function for readability
def chunk_end(chunk: pd.DataFrame) -> pd.Timestamp:
    """
    Function that gets the end time of a chunk.
    Args:
        chunk: pd.DataFrame - chunk of glucose data with datetime index
    Returns:
        pd.Timestamp - end time of the chunk
    """
    return chunk.index[-1]
