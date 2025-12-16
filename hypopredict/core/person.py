import os
import pandas as pd

import hypopredict.compressor as comp

class Person:

    def __init__(self, ID: int or str, ecg_dir: str = None) -> None:

        self.ID = int(ID)
        self.ecg = {}
        self.ecg_dir = ecg_dir

    def load_HG_data(self,
                     glucose_src: 'local' or 'gdrive',
                     min_duration: int = 15,
                     threshold: float = 3.9) -> None:
        """
        Load glucose data and identify HG events for the person.

        Args:
            glucose_src: str - source type ('local', 'gdrive', etc.)
            min_duration: int - minimum duration of HG event in minutes
        """
        self.glucose_src = glucose_src
        if self.glucose_src == "gdrive":
            self.glucose_data = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[self.ID-1])

        if self.glucose_src == "local":
            # GLUCOSE_PATH = '/Users/alexxela/code/hypopredict/data/dbt-glucose/'
            # load it from .env file
            GLUCOSE_PATH = os.getenv('GLUCOSE_PATH')
            # join with formatted filename
            GLUCOSE_PATH_ID = os.path.join(GLUCOSE_PATH, f'glucose_person{self.ID}.feather')

            self.glucose_data = pd.read_feather(GLUCOSE_PATH_ID)

        # identify hg events for that person
        self.hg_events = comp.identify_hg_events(
            self.glucose_data, min_duration=min_duration, threshold=threshold
        )

        del self.glucose_data

    def load_ECG_day(self, day: int or str, warning: bool = False) -> None:
        """
        Load and concatenate ECG data for a given day.

        Args:
            day: int, day identifier 1 through 4
            ecg_dir: str, directory where ECG data files are stored

        Creates:
            self.ecg[day]: pd.DataFrame, concatenated ECG data for the specified day
            self.ecg_dir: str, directory where ECG data files are stored
        """

        day = int(day)

        # concatinate all ecg files for that day
        self.ecg[day] = pd.DataFrame()
        ecg_day_paths = self._ECG_id_day_paths(day)

        if len(ecg_day_paths) > 1 and warning:
            print(
                f"""
    WARNING: there were multiple files for day _{self.ID}{day}_ => there might be a gap in concatinated ecg index so when you check if HG events actually ahppened during recorded ECG times check for this gap
    Files concatinated:
                """,
                ecg_day_paths
            )

        for path in ecg_day_paths:
            ECG_id_day_file = pd.read_feather(path)
            self.ecg[day] = pd.concat([self.ecg[day], ECG_id_day_file])


    def _ECG_id_day_paths(self, day: int) -> list:
        """
        Load all ECG feather file paths for a given IDDAY
        Args:
            day: int, day identifier (e.g. 73 = person 7 day 3)
        Returns:
            f_paths: list of file paths for ECG data for that day
        """

        f_paths = []
        for root, dirs, files in os.walk(self.ecg_dir):
            for file in files:
                if file.startswith(f"EcgWaveform-{self.ID}{day}"):
                    f_paths.append(os.path.join(root, file))

        return f_paths
