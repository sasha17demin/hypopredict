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
    """

    def __init__(self, days: list,
                 n_splits: int = 5,
                 random_state: int = 17):

        """
        days: a list of days identified by person, i.e. 73 = person 7 day 3
        n_splits: number of cross-validation splits. Total number of days should divisible by n_splits
        """

        self.days = np.array(sorted(days))
        self.n_splits = n_splits
        self.random_state = random_state

        self.people = np.unique(
                        np.array([int(day) // 10 for day in self.days])
                        ) # deduce unique people IDs from days

        #self.fold_size = np.ceil(len(days)/n_splits).astype(int)




    def get_splits(self):
        np.random.seed(self.random_state)

        #TODO: better way to shuffle
        shuffled_days = np.random.choice(self.days,
                                         size = self.days.size,
                                         replace=False)

        splits = np.array_split(shuffled_days, self.n_splits)

        return np.array(splits)






    def validate(self, splits, verbose=False):
        """
        Ensure each split has HG event, i.e. mean(is_HG) > 0
        """

        checks = []
        props = []
        for split in splits:

            split_hg_prop = np.mean(list(map(
                                lambda day: self._HG_prop_with_ECG(day), split
                                ))).round(4)
            props.append(split_hg_prop)

            res = split_hg_prop > 0
            checks.append(res)

            if verbose:
                if split_hg_prop > 0:
                    print(f'\nSplit is valid with {split_hg_prop*100}% of y == 1')
                else:
                    print('\nINVALID: no y == 1 in this split')

        return checks, props






    def _HG_prop_with_ECG(self, day, verbose = False):

        ID = int(day//10)

        #SIGNAL_TYPE = "EcgWaveform"
        #RAW_DATA_DIR = '../data/feathers'
        GLUCOSE_PATH = f'../../data/dbt-glucose/glucose_person{ID}.feather'


        person = {'ID': ID}

        #person['glucose'] = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[ID-1])
        person['glucose'] = pd.read_feather(GLUCOSE_PATH)
        person['hg_events'] = comp.identify_hg_events(person['glucose'], min_duration=15, threshold=3.9)




        ecg_day = f'ecg_{str(day)[1]}'
        person[ecg_day] = pd.DataFrame()
        ecg_day_paths = self._load_day(day)

        if len(ecg_day_paths) > 1:
            print(
                """
                WARNING: there were multiple files for 1 day
                        => there might be a gap in concatinated ecg index
                        so when you check
                        if HG events actually ahppened during recorded ECG times
                        check for this gap

                Files concatinated:
                """, ecg_day_paths
            )

        for path in ecg_day_paths:
            df = pd.read_feather(path)
            person[ecg_day] = pd.concat([person[ecg_day], df])

        hg_events_w_ecg = person['hg_events'].loc[
                        person[ecg_day].index.min() : person[ecg_day].index.max()
                        ]
###############################
        # TODO: identified days with no glucose measures for ECG
        if hg_events_w_ecg.size == 0:

            hg_prop_with_ecg = -1

        else:
            hg_prop_with_ecg = np.mean(hg_events_w_ecg['is_hg'] == 1)




        if verbose:
            print('\nProportion of HG glucose level among ALL glucose samples for this person-day')
            print(np.mean(person['hg_events']['is_hg'] == 1).round(2))
            print('''
        Proportion of HG glucose level among SUBSET glucose samples
            that have corresponding ECG records --> only these are useful for analysis''')
            print(round(hg_prop_with_ecg, 2))

        return hg_prop_with_ecg







    def _load_day(self, day):

        f_paths = []
        for root, dirs, files in os.walk(f'../../data/feathers'):
            for file in files:
                if file.startswith(f'EcgWaveform-{str(day)}'):
                    f_paths.append(os.path.join(root, file))

        return f_paths
