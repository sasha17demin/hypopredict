"""
Module with custom Cross-Validation split which keeps necessary within-day time patterns
while shuffling days within people
(since we are gonna be testing on just one day per person
and our model should generalize across days)
and shuffling between people
"""

import pandas as pd
import numpy as np
import os

from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import average_precision_score


from hypopredict import chunker, labeler
import hypopredict.compressor as comp
from hypopredict.core.person import Person
import hypopredict.chunk_preproc as cp
from hypopredict.new_features import prepare_X_y


class CV_splitter:
    """
    Class that handles train-validation splitting for cross-validation.
    Methods:
        - get_splits: returns list of splits (each split is a list of days)
        - validate: checks if each split has at least one hypoglycemic event and
                    returns list of booleans indicating validity of each split
                    as well as list of proportions of HG events in each split
    """

    def __init__(
        self,
        ecg_dir: str,
        glucose_src: str = "local" or "gdrive",
        n_splits: int = 5,
        random_state: int = 17,
    ):
        """

        n_splits: number of cross-validation splits. Total number of days should divisible by n_splits
        """

        self.n_splits = n_splits
        self.random_state = random_state
        self.ecg_dir = ecg_dir
        self.glucose_src = glucose_src

        if self.glucose_src == "local":
            assert (
                os.getenv("GLUCOSE_PATH") is not None
            ), "Set GLUCOSE_PATH environment variable for local glucose data"

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
        self.people = np.unique(np.array([int(day) // 10 for day in self.days]))

        np.random.seed(self.random_state)

        shuffled_days = np.random.permutation(self.days)

        splits = np.array_split(shuffled_days, self.n_splits)

        return np.array(splits)

    def validate(
        self, splits: np.ndarray, verbose: bool = False, warning: bool = False
    ) -> tuple:
        """
        Ensure each split has HG event, i.e. mean(is_HG) > 0
        Args:
            splits: np.ndarray of shape (n_splits, split_size)
                    each row is a split containing days
            verbose: whether to print validation logs
            warning: whether to print warnings for ECG days in multiple files
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
            split_HG_by_day = list(
                map(lambda day: self._HG_prop_with_ECG(day, warning=warning), split)
            )
            # then take mean proportion across days in split
            split_hg_prop = np.mean(split_HG_by_day).round(4)
            props.append(split_hg_prop)

            # boolean result of validation check
            res = split_hg_prop > 0
            checks.append(res)

            if verbose:
                if split_hg_prop > 0:
                    print(
                        f"\nSplit {split} is valid with {split_hg_prop*100}% of y == 1\n"
                    )
                elif split_hg_prop == 0:
                    print(
                        f"\nZERO SPLIT {split}: zero HG events, no y == 1 in this split\n"
                    )
                #########################
                # TODO: handle this case in training? resample splits?
                else:
                    # find index of day with -100.0 proportion
                    invalid_day_idx = split_HG_by_day.index(-100.0)
                    invalid_day = split[invalid_day_idx]
                    print(
                        f"\nINVALID SPLIT {split}: missing glucose measures for ECG on day {invalid_day}\n"
                    )

        return checks, np.array(props)

    def _HG_prop_with_ECG(
        self, day: int, verbose: bool = False, warning: bool = False
    ) -> float:
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

        ID = str(day)[0]  # person ID is first digit of day identifier
        person = Person(ID, ecg_dir=self.ecg_dir)

        # SIGNAL_TYPE = "EcgWaveform"
        # RAW_DATA_DIR = '../data/feathers'

        person.load_HG_data(glucose_src=self.glucose_src)

        # second number in day is day of recording for that person
        day_str = str(day)[1]
        person.load_ECG_day(day_str, warning=warning)

        # subset hg_events to ECG start and end time
        ###############################
        # TODO: what if there are multiple files for that day? handle gaps?
        hg_events_w_ecg = person.hg_events.loc[
            person.ecg[int(day_str)].index.min() : person.ecg[int(day_str)].index.max()
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
            print(np.mean(person.hg_events["is_hg"] == 1).round(2))
            print(
                """
        Proportion of HG glucose level among SUBSET glucose samples
            that have corresponding ECG records --> only these are useful for analysis"""
            )
            print(round(hg_prop_with_ecg, 2))

        return hg_prop_with_ecg


class CrossValidator:
    """
    Class to perform cross-validation using CV_splitter
    """

    def __init__(self, splits: np.ndarray):
        self.splits = splits

    def chunkify_label_stack(
        self,
        chunk_size: pd.Timedelta,
        step_size: pd.Timedelta,
        ecg_dir: str,
        forecast_window: pd.Timedelta,
        roll_window_size: pd.Timedelta,
        roll_step_size: pd.Timedelta,
        suffix: str,
        glucose_src: "local" or "gdrive" = "gdrive",
        agg_funcs: list = ["mean", "std", "min", "max"],
    ):

        # turn splits into prepped (X, y) pairs
        splits_prepped = []
        # this way we only feature-engineer once per split, not once per CV iteration
        for i in range(len(self.splits)):
            ##############################
            SPLIT_INDEX = i
            ##############################
            # take the split
            SPLIT = self.splits[SPLIT_INDEX]
            # chunkify
            split_chunkified = chunker.chunkify(
                SPLIT.ravel(),
                chunk_size=chunk_size,
                step_size=step_size,
                ecg_dir=ecg_dir,
            )
            # label chunks
            split_labels = labeler.label_split(
                split_chunkified,
                glucose_src=glucose_src,
                forecast_window=forecast_window,
            )
            # validate and stack
            chunks_split, y_split = cp.filter_and_stack(split_chunkified, split_labels)
            # prepare X, y with rolling features
            ########################
            # TODO: make prepare_X_y part of model class and argument here
            # that should be model's @classmethod preprocess
            X_split, y_split = prepare_X_y(
                chunks_split,
                y_split,
                roll_window_size=roll_window_size,
                roll_step_size=roll_step_size,
                suffix=suffix,
                agg_funcs=agg_funcs,
            )  # type: ignore ################ TODO: suffix
            # save for iterative CV
            splits_prepped.append((X_split, y_split))

        self.splits_prepped = splits_prepped

        return splits_prepped

    def _get_split_mean_labels(self, splits_prepped: list) -> list:
        extr = lambda x: np.mean(splits_prepped[x][1]).round(4)
        n_splits = len(splits_prepped)
        return list(map(extr, range(n_splits)))

    def validate_model_cv_VERBOSE(
        self,
        model,
        splits_prepped: list,
        resample: bool = True,
        desired_pos_ratio: float = 0.3,
        reduction_factor: float = 0.777
    ) -> dict:
        """
        Perform cross-validation and collect validation PR-AUCs
        Args:
            model: a scikit-learn compatible model with fit and predict_proba methods
            splits_prepped: list of (X, y) tuples for each split
        Returns:
            val_pr_aucs: list of validation PR-AUCs for each split
            val_ave_precisions: list of validation average precision scores for each split
        """

        # collect val PR-AUCs
        # from sklearn.metrics import precision_recall_curve, auc
        # collect val average precision scores
        # from sklearn.metrics import average_precision_score
        val_pr_aucs = []
        val_ave_precisions = []


        for VAL_SPLIT_INDEX in range(len(splits_prepped)):

            print("==================================================")
            print(
                f"Cross-Validation Iteration: Using split {VAL_SPLIT_INDEX} as validation set"
            )
            print(
                f"""
                        With mean positive class ratio: {self._get_split_mean_labels(splits_prepped)[VAL_SPLIT_INDEX]:.3f}\n
                """
            )

            X_val, y_val = splits_prepped[VAL_SPLIT_INDEX]

            train_splits_idx = [
                i for i in range(len(splits_prepped)) if i != VAL_SPLIT_INDEX
            ]

            if str(model.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>" and not resample:
                original_labels = np.concatenate([splits_prepped[i][1] for i in train_splits_idx])
                scale_pos_weight = (original_labels == 0).sum() / (original_labels == 1).sum()

                model.scale_pos_weight = scale_pos_weight
                print(f">>>>   Setting XGBoost scale_pos_weight to {scale_pos_weight:.3f}")

            splits_prepped_resampled = splits_prepped.copy()
            if resample:

                self.desired_pos_ratio = desired_pos_ratio
                self.reduction_factor = reduction_factor

                print("==================================================")
                print(
                    f"Resampling training folds {train_splits_idx} \n to achieve ~{self.desired_pos_ratio} positive class ratio..."
                )

                for SPLIT_INDEX in train_splits_idx:
                    splits_prepped_resampled[SPLIT_INDEX] = (
                        self._resample_split_recursive(
                            splits_prepped_resampled[SPLIT_INDEX],
                            desired_pos_ratio=self.desired_pos_ratio,
                            reduction_factor=self.reduction_factor,
                        )
                    )

                print("RESAMPLED")

            # stack X_trains from other splits
            X_train = pd.concat(
                [splits_prepped_resampled[i][0] for i in train_splits_idx]
            )
            y_train = np.hstack(
                [splits_prepped_resampled[i][1] for i in train_splits_idx]
            )

            print(f"Train positive class ratio: {np.mean(y_train):.3f}")

            # fit model
            print("--------------------------------------------------")
            print(f"Fitting model on training folds {train_splits_idx}...")



            model.fit(X_train, y_train)

            print(f"Evaluating model on training folds {train_splits_idx}...")
            # predict probabilities
            y_probs_train = model.predict_proba(X_train)[:, 1]

            # compute PR-AUC
            precision, recall, _ = precision_recall_curve(y_train, y_probs_train)
            pr_auc = auc(recall, precision)
            val_pr_aucs.append(round(pr_auc, 3))

            # compute average precision score
            ave_precision = average_precision_score(y_train, y_probs_train)
            val_ave_precisions.append(round(ave_precision, 3))
            print(
                f"""
                    TRAIN PR-AUC: {pr_auc:.4f}, Average Precision: {ave_precision:.4f}
                """
            )
            print("""
                ••••••••••••••••••••••••••••••••••••••••••••••••••••••••
                  """)


            print(f"Evaluating model on VALIDATION fold {VAL_SPLIT_INDEX}...")
            # predict probabilities
            y_probs_val = model.predict_proba(X_val)[:, 1]

            # compute PR-AUC
            precision, recall, _ = precision_recall_curve(y_val, y_probs_val)
            pr_auc = auc(recall, precision)
            val_pr_aucs.append(round(pr_auc, 3))

            # compute average precision score
            ave_precision = average_precision_score(y_val, y_probs_val)
            val_ave_precisions.append(round(ave_precision, 3))
            print(
                f"""
                    VALIDATION PR-AUC: {pr_auc:.4f}, Average Precision: {ave_precision:.4f}
                """
            )
            print("""
                ••••••••••••••••••••••••••••••••••••••••••••••••••••••••\n\n
                  """)
        return {"val_pr_aucs": val_pr_aucs, "val_ave_precisions": val_ave_precisions}

    def _resample_split_recursive(
        self, split, desired_pos_ratio=0.4, reduction_factor=0.5
    ):

        # recurrently check if we're above threshold
        if np.mean(split[1]) >= desired_pos_ratio \
                    and np.mean(split[1]) <= desired_pos_ratio + 0.1:
            return split

        # if there are too many ones, undersample positive class
        elif np.mean(split[1]) >= desired_pos_ratio + 0.1:
            ys = split[1]
            pos_idx = np.where(ys == 1)[0]
            neg_idx = np.where(ys == 0)[0]

            undersamp_idx = np.random.choice(
                pos_idx, int(pos_idx.size * reduction_factor), replace=False
            )
            new_idx = np.concatenate([undersamp_idx, neg_idx])
            new_idx.sort()

            X_split_resampled = split[0].iloc[new_idx]
            y_split_resampled = split[1][new_idx]

            split_resampled = (X_split_resampled, y_split_resampled)

            return self._resample_split_recursive(
                split_resampled, desired_pos_ratio, reduction_factor
            )

        # if there are too many zeros, undersample negative class
        #elif np.mean(split[1]) <= desired_pos_ratio:
        else:
            ys = split[1]
            pos_idx = np.where(ys == 1)[0]
            neg_idx = np.where(ys == 0)[0]

            undersamp_idx = np.random.choice(
                neg_idx, int(neg_idx.size * reduction_factor), replace=True
            )
            new_idx = np.concatenate([undersamp_idx, pos_idx])
            new_idx.sort()

            X_split_resampled = split[0].iloc[new_idx]
            y_split_resampled = split[1][new_idx]

            split_resampled = (X_split_resampled, y_split_resampled)

            return self._resample_split_recursive(
                split_resampled, desired_pos_ratio, reduction_factor
            )

    def validate_model_cv(
            self,
            model,
            splits_prepped: list,
            resample: bool = True,
            desired_pos_ratio: float = 0.3,
            reduction_factor: float = 0.777
        ) -> dict:
            """
            Perform cross-validation and collect validation PR-AUCs
            Args:
                model: a scikit-learn compatible model with fit and predict_proba methods
                splits_prepped: list of (X, y) tuples for each split
            Returns:
                val_pr_aucs: list of validation PR-AUCs for each split
                val_ave_precisions: list of validation average precision scores for each split
            """
            val_pr_aucs = []
            val_ave_precisions = []


            for VAL_SPLIT_INDEX in range(len(splits_prepped)):

                X_val, y_val = splits_prepped[VAL_SPLIT_INDEX]

                X_not_na_idx
                y_val = y_val[X_val.index]

                train_splits_idx = [
                    i for i in range(len(splits_prepped)) if i != VAL_SPLIT_INDEX
                ]

                if str(model.__class__) == "<class 'xgboost.sklearn.XGBClassifier'>" and not resample:
                    original_labels = np.concatenate([splits_prepped[i][1] for i in train_splits_idx])
                    scale_pos_weight = (original_labels == 0).sum() / (original_labels == 1).sum()

                    model.scale_pos_weight = scale_pos_weight

                splits_prepped_resampled = splits_prepped.copy()
                if resample:

                    self.desired_pos_ratio = desired_pos_ratio
                    self.reduction_factor = reduction_factor

                    for SPLIT_INDEX in train_splits_idx:
                        splits_prepped_resampled[SPLIT_INDEX] = (
                            self._resample_split_recursive(
                                splits_prepped_resampled[SPLIT_INDEX],
                                desired_pos_ratio=self.desired_pos_ratio,
                                reduction_factor=self.reduction_factor,
                            )
                        )


                # stack X_trains from other splits
                X_train = pd.concat(
                    [splits_prepped_resampled[i][0] for i in train_splits_idx]
                )
                y_train = np.hstack(
                    [splits_prepped_resampled[i][1] for i in train_splits_idx]
                )

                X_train.dropna(inplace=True)
                y_train = y_train[X_train.index]

                model.fit(X_train, y_train)

                # predict probabilities
                y_probs_train = model.predict_proba(X_train)[:, 1]

                # compute PR-AUC
                precision, recall, _ = precision_recall_curve(y_train, y_probs_train)
                pr_auc = auc(recall, precision)
                val_pr_aucs.append(round(pr_auc, 3))

                # compute average precision score
                ave_precision = average_precision_score(y_train, y_probs_train)
                val_ave_precisions.append(round(ave_precision, 3))

                # predict probabilities
                y_probs_val = model.predict_proba(X_val)[:, 1]

                # compute PR-AUC
                precision, recall, _ = precision_recall_curve(y_val, y_probs_val)
                pr_auc = auc(recall, precision)
                val_pr_aucs.append(round(pr_auc, 3))

                # compute average precision score
                ave_precision = average_precision_score(y_val, y_probs_val)
                val_ave_precisions.append(round(ave_precision, 3))

            return {"val_pr_aucs": val_pr_aucs, "val_ave_precisions": val_ave_precisions}
