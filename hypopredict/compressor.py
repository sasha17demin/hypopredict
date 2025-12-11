import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # suppress pandas warnings


# hardcoded google drive links to glucose data feather files
GLUCOSE_ID_LINKS = [
    "https://drive.google.com/file/d/1qGfSIb9EEJ4ZxlWnBcsILgh9LbHAiMld/view?usp=share_link",
    "https://drive.google.com/file/d/1T8vah3NvPtuBLenHCwhYsdHX7Ug7b84-/view?usp=share_link",
    "https://drive.google.com/file/d/1PDOqLYVxHuvfajuuLO9R-ul0288-PYuR/view?usp=share_link",
    "https://drive.google.com/file/d/18sLNjzhyCsSVHNa9UQCtpqwxlpen2rDB/view?usp=share_link",
    "https://drive.google.com/file/d/1lU0J_VstbwGIQ7PhKTj-Tn2Gq2ccXDiA/view?usp=share_link",
    "https://drive.google.com/file/d/1fAn4YpvyBDcVFlqUJEhhQP0zmrJ8fofk/view?usp=share_link",
    "https://drive.google.com/file/d/1G5y29odvmuShSlq2tGMGcCss8yo4OAjh/view?usp=share_link",
    "https://drive.google.com/file/d/1_oBIe_aT6k9EePdlS5ywUfxEKFuYM1y4/view?usp=share_link",
    "https://drive.google.com/file/d/1hOvQPVkJmAbkjqJcpbMJGXERcNiH4yAY/view?usp=share_link",
]



# read google drive link to pandas dataframe
def gdrive_to_pandas(link):
    """
    Function that reads a google drive link to a feather file
    (can be any feature but should .feather)
    and returns a pandas dataframe.

    Args:
        link: str - google drive link to the feather file
    Returns:
        pd.DataFrame - dataframe read from the feather file
    """

    file_id = link.split("/")[-2]
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    return pd.read_feather(url)


# identify HG events: <3.9 for at least 15 minutes
def identify_hg_events(
    glucose_df: pd.DataFrame, cgm_only: bool = True, threshold=3.9, min_duration=15
) -> pd.DataFrame:
    """Identify hypoglycemic events in glucose data.
    Args:
        glucose_df (pd.DataFrame): DataFrame with 'timestamp' and 'glucose' columns.
        threshold (float): Glucose level threshold for hypoglycemia.
        min_duration (int): Minimum duration (in minutes) to consider as an event.
    Returns:
        pd.DataFrame: DataFrame with identified hypoglycemic events.
                    Columns = [start_time, end_time, duration, min_glucose]
    """
    if cgm_only:
        glucose_df = glucose_df[glucose_df["type"] == "cgm"]

    glucose_df["is_hg"] = (glucose_df["glucose"] < threshold).astype(int)
    # if less than min_duration consecutive readings below threshold, set to 0
    glucose_df["hg_group"] = (
        glucose_df["is_hg"] != glucose_df["is_hg"].shift()
    ).cumsum()
    hg_durations = glucose_df.groupby("hg_group")["is_hg"].transform("sum")
    glucose_df.loc[
        (glucose_df["is_hg"] == 1) & (hg_durations < (min_duration // 5)), "is_hg"
    ] = 0
    glucose_df.drop(columns=["hg_group"], inplace=True)
    # record the onsets
    glucose_df["onset"] = (glucose_df["is_hg"] > glucose_df["is_hg"].shift()).astype(
        int
    )  # if now 1 but was 0 before
    glucose_df["end"] = (glucose_df["is_hg"].shift() > glucose_df["is_hg"]).astype(int)

    return glucose_df


# plot glucose level and HG events
def plot_hg_events(person):
    """
    Function that plots glucose level and HG events for a person.
    Args:
        person: dict - dictionary with keys 'ID', 'glucose', 'hg_events'
    Returns:
        None - shows a plot
    """

    plt.figure(figsize=(15, 5))
    plt.plot(
        person["hg_events"].index, person["hg_events"]["glucose"], label="Glucose Level"
    )
    plt.plot(
        person["hg_events"].index, person["hg_events"]["is_hg"], label="HG indicator"
    )
    # add vertical lines for onsets and ends
    for idx, row in person["hg_events"].iterrows():
        if row["onset"] == 1:
            plt.axvline(
                x=idx,
                color="red",
                linestyle="--",
                label=(
                    "HG Onset"
                    if "HG Onset" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
        if row["end"] == 1:
            plt.axvline(
                x=idx,
                color="green",
                linestyle="--",
                label=(
                    "HG End"
                    if "HG End" not in plt.gca().get_legend_handles_labels()[1]
                    else ""
                ),
            )
    plt.axhline(y=3.9, color="yellow", linestyle=":", label="HG Threshold (3.9 mmol/L)")
    plt.xlabel("Time")
    plt.ylabel("Glucose Level (mmol/L)")
    plt.title(
        f'Person {person["ID"]} Glucose Level with Hypoglycemic Events < 3.9 mmol/L for at Least 15 Minutes'
    )
    plt.legend()
    plt.show()


# count number of HG events per day
def day_count_HG(person):
    """
    Function to count the number of hypoglycemic events per day for a person.
    Args:
        person: dict - dictionary with keys 'ID', 'glucose', and 'hg_events'
    Returns:
        dict - dictionary with days as keys and event counts as values
    """

    # get index of HG onsets
    onset_times = person["hg_events"].index[person["hg_events"]["onset"] == 1]

    res = dict(zip(["1", "2", "3", "4", "5", "6"], [0] * 6))

    for onset in onset_times:
        day = str(onset.day)
        if day in res:
            res[day] += 1
    return res


# parse sensor CSV into pandas, compress numeric values, and index by datetime
def parse_compress_csv(file_path, signal_type):
    """
    Parses a CSV file, compresses the specified signal type to uint16,
    and indexes the DataFrame by datetime.
    Args:
        file_path: str - path to the CSV file
        signal_type: str - _column name_ in .csv of the signal to compress
    Returns:
        pd.DataFrame - processed DataFrame
    """

    print(f"Processing file: {file_path}")
    csv_size = os.path.getsize(file_path) / 1024**2
    print(f"Original CSV file size: {csv_size:.2f} MB")

    # Read and preprocess the CSV file
    df = pd.read_csv(file_path, dtype={signal_type: "uint16"})

    df["datetime"] = pd.to_datetime(
        df["Time"], dayfirst=True, format="%d/%m/%Y %H:%M:%S.%f"
    )

    df.set_index("datetime", inplace=True)
    df.drop(columns=["Time"], inplace=True)

    print(df.head())
    return df


# save processed DataFrame to feather file
def save_to_feather(df, new_feather_path):
    """
    Saves the DataFrame to a feather file and prints the file size.
    Args:
        df: pd.DataFrame - DataFrame to save
        new_feather_path: str - path to save the feather file
    Returns:
        None
    """

    # Save to feather format
    df.to_feather(new_feather_path)

    feather_size = os.path.getsize(new_feather_path) / 1024**2
    print(f"Saved to feather: {new_feather_path} ({feather_size:.2f} MB)")
