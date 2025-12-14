"""
Person class for managing per-person glucose and ECG data.

Integrated with PersonDay for type-safe day management and label handling.
"""
import os
import pandas as pd
from typing import Optional, Union

from hypopredict.data import loaders, labels
from hypopredict.core.data_types import PersonDay


class Person:
    """
    Represents a single person with their glucose and ECG data.
    
    Attributes:
        ID: Integer person identifier (1-9)
        ecg: Dictionary mapping day number to ECG DataFrame
        ecg_dir: Directory path for ECG data files
        glucose_data: Raw glucose measurements (if loaded)
        hg_events: DataFrame with hypoglycemia event annotations (if loaded)
    """

    def __init__(self, ID: Union[int, str], ecg_dir: Optional[str] = None) -> None:
        """
        Initialize Person object.
        
        Args:
            ID: Person identifier (1-9)
            ecg_dir: Optional directory path for ECG data files
        """
        self.ID = int(ID)
        self.ecg = {}
        self.ecg_dir = ecg_dir
        self.glucose_data = None
        self.hg_events = None
        self.glucose_src = None

    def load_HG_data(
        self,
        glucose_src: str = 'local',
        min_duration: int = 15,
        threshold: float = 3.9
    ) -> None:
        """
        Load glucose data and identify HG events for the person.

        Args:
            glucose_src: Source type ('local' or 'gdrive')
            min_duration: Minimum duration of HG event in minutes
            threshold: Glucose threshold for hypoglycemia (mmol/L)
        """
        self.glucose_src = glucose_src
        
        if self.glucose_src == "gdrive":
            self.glucose_data = loaders.gdrive_to_pandas(
                loaders.GLUCOSE_ID_LINKS[self.ID - 1]
            )

        if self.glucose_src == "local":
            # Load from local path specified in environment
            GLUCOSE_PATH = os.getenv('GLUCOSE_PATH')
            if GLUCOSE_PATH is None:
                raise ValueError(
                    "GLUCOSE_PATH environment variable not set. "
                    "Please set it to the directory containing glucose data."
                )
            GLUCOSE_PATH_ID = os.path.join(
                GLUCOSE_PATH, 
                f'glucose_person{self.ID}.feather'
            )
            self.glucose_data = pd.read_feather(GLUCOSE_PATH_ID)

        # Identify HG events for this person
        self.hg_events = labels.identify_hg_events(
            self.glucose_data, 
            min_duration=min_duration, 
            threshold=threshold
        )

        # Free up memory by deleting raw glucose data
        del self.glucose_data
        self.glucose_data = None

    def load_ECG_day(
        self, 
        day: Union[int, str, PersonDay], 
        warning: bool = True
    ) -> None:
        """
        Load and concatenate ECG data for a given day.

        Args:
            day: Day identifier (1-6) or PersonDay object
            warning: Whether to print warnings for multiple file gaps

        Creates:
            self.ecg[day]: pd.DataFrame with concatenated ECG data for the day
        """
        # Handle PersonDay object
        if isinstance(day, PersonDay):
            if day.person_id != self.ID:
                raise ValueError(
                    f"PersonDay person_id ({day.person_id}) doesn't match "
                    f"Person ID ({self.ID})"
                )
            day = day.day
        
        day = int(day)

        # Concatenate all ECG files for that day
        self.ecg[day] = pd.DataFrame()
        ecg_day_paths = self._ECG_id_day_paths(day)

        if len(ecg_day_paths) > 1 and warning:
            print(
                f"""
    WARNING: there were multiple files for day _{self.ID}{day}_ => there might be a gap in concatenated ecg index so when you check if HG events actually happened during recorded ECG times check for this gap
    Files concatenated:
                """,
                ecg_day_paths
            )

        for path in ecg_day_paths:
            ECG_id_day_file = pd.read_feather(path)
            self.ecg[day] = pd.concat([self.ecg[day], ECG_id_day_file])

    def _ECG_id_day_paths(self, day: int) -> list:
        """
        Load all ECG feather file paths for a given person-day.
        
        Args:
            day: Day identifier (1-6)
            
        Returns:
            List of file paths for ECG data for that day
        """
        if self.ecg_dir is None:
            raise ValueError(
                "ecg_dir not set. Please provide ecg_dir during Person initialization."
            )

        f_paths = []
        for root, dirs, files in os.walk(self.ecg_dir):
            for file in files:
                if file.startswith(f"EcgWaveform-{self.ID}{day}"):
                    f_paths.append(os.path.join(root, file))

        return f_paths
    
    def get_HG_onset_times_for_day(
        self, 
        day: Union[int, PersonDay], 
        threshold: float = 3.9
    ) -> list:
        """
        Get hypoglycemia onset times for a specific day.
        
        Args:
            day: Day identifier (1-6) or PersonDay object
            threshold: Glucose threshold for hypoglycemia (mmol/L)
            
        Returns:
            List of onset timestamps for that day
        """
        # Handle PersonDay object
        if isinstance(day, PersonDay):
            if day.person_id != self.ID:
                raise ValueError(
                    f"PersonDay person_id ({day.person_id}) doesn't match "
                    f"Person ID ({self.ID})"
                )
            day = day.day
        
        day = int(day)
        
        if self.hg_events is None:
            raise ValueError(
                "HG events not loaded. Call load_HG_data() first."
            )
        
        # Filter hg_events to the specific day
        day_events = self.hg_events[self.hg_events.index.day == day]
        
        # Get onset times
        onset_times = day_events[day_events['onset'] == 1].index.tolist()
        
        return onset_times
