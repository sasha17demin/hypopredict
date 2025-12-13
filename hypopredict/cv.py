"""
Module with custom Cross-Validation split which keeps necessary within-day time patterns
while shuffling days within people
(since we are gonna be testing on just one day per person
and our model should generalize across days)
and shuffling between people
"""

import pandas as pd
import numpy as np

import hypopredict.compressor as comp

import os


class CV_splitter:
    """
    Class that handles train-validation splitting for cross-validation.
    Methods:
        - get_splits: returns list of splits (each split is a list of days)
        - validate: checks if each split has at least one hypoglycemic event and
                    returns list of booleans indicating validity of each split
                    as well as list of proportions of HG events in each split
    """

    def __init__(self,
                 ecg_dir: str,
                 glucose_src: str = "local" or "gdrive",
                 n_splits: int = 5, random_state: int = 17):
        """

        n_splits: number of cross-validation splits. Total number of days should divisible by n_splits
        """


        self.n_splits = n_splits
        self.random_state = random_state
        self.ecg_dir = ecg_dir
        self.glucose_src = glucose_src

        # self.fold_size = np.ceil(len(days)/n_splits).astype(int)




    def get_splits(self, days: list) -> np.ndarray:
        """
        Generate random splits of days for cross-validation
        Args:
            days: a list of days identified by person, i.e. 73 = person 7 day 3
        Returns:
            splits: np.ndarray of shape (n_splits, split_size)
                    each row is a split containing days
        """

        self.days = np.array(sorted(days))
        # deduce unique people IDs from days
        self.people = np.unique(
            np.array([int(day) // 10 for day in self.days])
        )

        np.random.seed(self.random_state)

        shuffled_days = np.random.permutation(self.days)

        splits = np.array_split(shuffled_days, self.n_splits)

        return splits

    def validate(self, splits: np.ndarray, verbose: bool = False) -> tuple:
        """
        Ensure each split has HG event, i.e. mean(is_HG) > 0
        Args:
            splits: np.ndarray of shape (n_splits, split_size)
                    each row is a split containing days
            verbose: whether to print validation logs
        Returns:
            checks: list of booleans indicating whether each split is valid
            props: list of proportions of HG events in each split
        """

        checks = []
        props = []
        for split in splits:
            print("--------------------------------------------------")
            print(f"Validating split: {split}")

            # within split, apply _HG_prop_with_ECG to each day
            split_HG_by_day = list(map(lambda day: self._HG_prop_with_ECG(day), split))
            # then take mean proportion across days in split
            split_hg_prop = np.mean(split_HG_by_day).round(4)
            props.append(split_hg_prop)

            # boolean result of validation check
            res = split_hg_prop > 0
            checks.append(res)

            if verbose:
                if split_hg_prop > 0:
                    print(f"\nSplit {split} is valid with {split_hg_prop*100}% of y == 1\n")
                elif split_hg_prop == 0:
                    print(f"\nZERO SPLIT {split}: zero HG events, no y == 1 in this split\n")
                else:
                    # find index of day with -100.0 proportion
                    invalid_day_idx = split_HG_by_day.index(-100.0)
                    invalid_day = split[invalid_day_idx]
                    print(f"\nINVALID SPLIT {split}: missing glucose measures for ECG on day {invalid_day}\n")

        return checks, props

    def _HG_prop_with_ECG(self, day: int,
                          verbose: bool = False) -> float:
        """
        For a given day, compute proportion of hypoglycemic events
        among glucose samples that have corresponding ECG records
        Args:
            day: int, day identifier (e.g. 73 = person 7 day 3)
            verbose: whether to print logs
        Returns:
            hg_prop_with_ecg: float, proportion of hypoglycemic events
                              among glucose samples that have corresponding ECG records
        """

        ID = int(day // 10)
        person = {"ID": ID}

        # SIGNAL_TYPE = "EcgWaveform"
        # RAW_DATA_DIR = '../data/feathers'



        if self.glucose_src == "gdrive":
            person['glucose'] = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[ID-1])

        if self.glucose_src == "local":
            # GLUCOSE_PATH = '/Users/alexxela/code/hypopredict/data/dbt-glucose/'
            # load it from .env file
            GLUCOSE_PATH = os.getenv('GLUCOSE_PATH')
            # join with formatted filename
            GLUCOSE_PATH_ID = os.path.join(GLUCOSE_PATH, f'glucose_person{ID}.feather')

            person["glucose"] = pd.read_feather(GLUCOSE_PATH_ID)

        # identify hg events for that person
        person["hg_events"] = comp.identify_hg_events(
            person["glucose"], min_duration=15, threshold=3.9
        )

        # create a key for ecg of that day
        ecg_day = f"ecg_{str(day)[1]}" # ecg_3 for day 73
        # concatinate all ecg files for that day
        person[ecg_day] = pd.DataFrame()
        ecg_day_paths = self._load_day(day)

        if len(ecg_day_paths) > 1:
            print(
                f"""
    WARNING: there were multiple files for day _{day}_ => there might be a gap in concatinated ecg index so when you check if HG events actually ahppened during recorded ECG times check for this gap
    Files concatinated:
                """,
                ecg_day_paths
            )

        for path in ecg_day_paths:
            df = pd.read_feather(path)
            person[ecg_day] = pd.concat([person[ecg_day], df])

        # subset hg_events to ECG start and end time
###############################
# TODO: what if there are multiple files for that day? handle gaps?
        hg_events_w_ecg = person["hg_events"].loc[
            person[ecg_day].index.min() : person[ecg_day].index.max()
        ]
###############################
# TODO: identified days with no glucose measures for ECG
        if hg_events_w_ecg.size == 0:

            hg_prop_with_ecg = -100.0  # indicate no glucose measures for that day

        else:
            hg_prop_with_ecg = np.mean(hg_events_w_ecg["is_hg"] == 1)

        if verbose:
            print(
                "\nProportion of HG glucose level among ALL glucose samples for this person-day"
            )
            print(np.mean(person["hg_events"]["is_hg"] == 1).round(2))
            print(
                """
        Proportion of HG glucose level among SUBSET glucose samples
            that have corresponding ECG records --> only these are useful for analysis"""
            )
            print(round(hg_prop_with_ecg, 2))

        return hg_prop_with_ecg


    def _load_day(self, day: int) -> list:
        """
        Load all ECG feather file paths for a given day
        Args:
            day: int, day identifier (e.g. 73 = person 7 day 3)
        Returns:
            f_paths: list of file paths for ECG data for that day
        """

        f_paths = []
        for root, dirs, files in os.walk(self.ecg_dir):
            for file in files:
                if file.startswith(f"EcgWaveform-{str(day)}"):
                    f_paths.append(os.path.join(root, file))

        return f_paths
