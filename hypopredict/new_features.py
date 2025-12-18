import pandas as pd
import numpy as np


# def function that extracts features, repeats relevant y_train, and stacks
def prepare_X_y(chunks,
                labels,
                roll_window_size,
                roll_step_size,
                suffix,
                agg_funcs=['mean', 'std', 'min', 'max']):
    X_list = []
    y_array = np.array([])
    time_list = []

    for i in range(len(chunks)):
        chunk = chunks[i]
        label = labels[i]
        X_feat = extract_rolling_features(
            chunk,
            roll_window_size = roll_window_size,
            roll_step_size=roll_step_size,
            suffix=suffix,
            agg_funcs=agg_funcs
        )
        # repeat y_train to match length of X_train_train
        repeat_factor = X_feat.shape[0]
        y_repeated = label.repeat(repeat_factor)
# TODO: add column with chunk_last_index in total seconds repeated
        forecast_index = chunk.index[-1]

        # Or get total seconds since midnight
        seconds_since_midnight = np.int64(forecast_index.hour * 3600 + forecast_index.minute * 60 + forecast_index.second)

        X_feat['timestamp'] = [forecast_index]*repeat_factor
        X_feat['timeofday'] = seconds_since_midnight.repeat(repeat_factor)

        X_list.append(X_feat)
        y_array = np.append(y_array, y_repeated)

    X_stacked = pd.concat(X_list, ignore_index=True)

    return X_stacked, y_array



# feature engineer on all splits separately, then stack for CV
def extract_rolling_features(chunk: pd.DataFrame,
                             roll_window_size: pd.Timedelta,
                             suffix: str,
                             roll_step_size: pd.Timedelta,
                             column_name:str = 'EcgWaveform',
                             agg_funcs: list = ['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis']):
    """
    Extract rolling window features with custom step size.
    Args:
        chunk: pandas df (ECG data)
        column_name: name of the column to compute features on
        window_size: window length in pd.Timedelta (e.g., pd.Timedelta(seconds=60))
        suffix: suffix to append to feature names (e.g., 'rolling60s')
        step_size: step length in pd.Timedelta (e.g., pd.Timedelta(seconds=1))
        agg_funcs: list of aggregation functions
    Returns:
        DataFrame with features, one row per step
    """

    # convert window_size and step_size from Timedelta to number of samples
    sampling_rate = int(chunk.resample('1s').count().mode().values[0][0])

    roll_window_size = int(roll_window_size.total_seconds() * sampling_rate)

    roll_step_size = int(roll_step_size.total_seconds() * sampling_rate)

    results = []
 #   indices = []
    # Slide window with step_size

    if roll_step_size <= 0:
        roll_step_size = 1  # ensure at least a step of 1
        #print(f"Warning: for chunk{i-1} step_size was less than or equal to 0. Set to 1 sample.")

    for start_idx in range(0, chunk.shape[0] - roll_window_size + 1, roll_step_size):
        end_idx = start_idx + roll_window_size
        window_data = chunk.iloc[start_idx:end_idx]
        # Compute all aggregations for this window
        agg_dict = {f'{suffix}_{func}': getattr(window_data[column_name], func)() for func in agg_funcs}
        results.append(agg_dict)
#        indices.append(start_idx)
    # Build DataFrame
    features = pd.DataFrame(results)
    return features
