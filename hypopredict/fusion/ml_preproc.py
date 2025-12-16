"""
Preprocesses raw data for SVM, XGBoost, and KNN base models.
Handles chunking, labeling, validation, stacking.
Then creates rolling features within chunks and stacks them into tables.
Allows separation between .fit() and .transform() for train and test data.
"""

import numpy as np
import pandas as pd
import os

from hypopredict.cv import CV_splitter
from hypopredict import chunker
from hypopredict import labeler

from hypopredict.cv import CrossValidator

from hypopredict.params import mlpreproc_params


class MLPreprocessor:

    def __init__(self):
        self.params = mlpreproc_params

        # np.random.seed(self.params['random_state'])

    def set_params(self, **kwargs) -> None:
        """
        Set preprocessing parameters.

        Args:
            params: dict - dictionary of preprocessing parameters
        """
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
            else:
                raise KeyError(f"Invalid parameter: {key}")

    def reset_params(self) -> None:
        """
        Reset preprocessing parameters to default values.
        """
        # reuse __init__ to reset params

        self.params = mlpreproc_params


    def split(self, days: list) -> np.ndarray:
        """
        Split data into training and testing sets based on specified days.

        Args:
            days: list - list of days to split on

        Returns:
            np.ndarray - array of CV-ready splits (IDDAYS)
        """

        # np.random.seed(self.params['random_state'])

        self.splitter = CV_splitter(
            n_splits=self.params["n_splits"],
            ecg_dir=self.params["ecg_dir"],
            glucose_src=self.params["glucose_src"],
            random_state=self.params["random_state"],
        )
        # get splits
        self.splits = self.splitter.get_splits(days)

        # return self.splits

# TODO: if test, should just take list of days and return preprocessed test (X,y)
    def preprocess(self) -> None:
        """
        Preprocess the data by chunking, labeling, stacking, and feature engineering.
        Uses parameters set in MLPreprocessor.params.

        Creates:
            self.splits_prepped: list - list of preprocessed (X, y) splits
        """

        # np.random.seed(self.params['random_state'])

        self.crossval = CrossValidator(splits=self.splits)
        # ectract chunkify_label_stack params from self.params

        chunkify_label_stack_params = {
            key: self.params[key]
            for key in [
                "chunk_size",
                "step_size",
                "ecg_dir",
                "glucose_src",
                "forecast_window",
                "roll_window_size",
                "roll_step_size",
                "suffix",
                "agg_funcs",
            ]
        }
        self.splits_prepped = self.crossval.chunkify_label_stack(
            **chunkify_label_stack_params
        )

    # save preprocessed splits and self.params that generated it
    # into one local file for easy read later
    def save_prepped_splits(self, filepath: str) -> None:
        """
        Save preprocessed splits and parameters to a local file.

        Args:
            filepath: str - path to save the preprocessed splits
        """
        import pickle

        dict_to_save = {"splits_prepped": self.splits_prepped, "params": self.params}

        with open(filepath, "wb") as f:
            pickle.dump(dict_to_save, f)

    # use this separately to load preprocessed splits and params
    # to feed into model training/evaluation
    @classmethod
    def load_prepped_splits(cls, filepath: str) -> dict:
        """
        Load preprocessed splits and parameters from a local file.

        Args:
            filepath: str - path to load the preprocessed splits from
        Returns:
            dict - dictionary containing 'splits_prepped' (X, y) and 'params'
        """
        import pickle

        with open(filepath, "rb") as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
